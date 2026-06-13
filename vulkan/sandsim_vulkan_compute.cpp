#include <SDL2/SDL.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static void vkCheck(VkResult result, const char *msg) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(msg);
  }
}

static std::vector<char> readFile(const char *path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open file");
  size_t size = (size_t)file.tellg();
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), size);
  return buffer;
}

class VulkanSandSim {
public:
  static constexpr int PIXEL_SIZE = 2;
  VulkanSandSim(int w, int h, bool headless = false)
      : width(w), height(h), renderWidth(w * PIXEL_SIZE),
        renderHeight(h * PIXEL_SIZE), rng(std::random_device{}()) {
    if (!headless)
      initSDL();
    initVulkan();
    createBuffers();
    createDescriptorSetLayout();
    createPipeline();
    createDescriptorPoolAndSet();
    createCommandPoolAndBuffer();
  }
  ~VulkanSandSim() {
    vkDeviceWaitIdle(device);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
    vkDestroyBuffer(device, storageBuffer, nullptr);
    vkFreeMemory(device, storageMemory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  void run() {
    bool quit = false;
    SDL_Event event;
    bool mouseDown = false;
    int mouseX = 0, mouseY = 0;
    int srcIndex = 0;
    int dstIndex = 1;
    std::memset(mappedPtr, 0, bufferBytes);
    while (!quit) {
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
          quit = true;
        else if (event.type == SDL_MOUSEBUTTONDOWN)
          mouseDown = true;
        else if (event.type == SDL_MOUSEBUTTONUP)
          mouseDown = false;
        else if (event.type == SDL_MOUSEMOTION)
          SDL_GetMouseState(&mouseX, &mouseY);
        else if (event.type == SDL_KEYDOWN) {
          if (event.key.keysym.sym == SDLK_c)
            std::memset(mappedPtr, 0, bufferBytes);
          else if (event.key.keysym.sym == SDLK_r)
            randomizeGrid(srcIndex);
        }
      }
      if (mouseDown)
        addSandCPU(srcIndex, mouseX, mouseY, 5);
      copyGrid(srcIndex, dstIndex);
      recordAndSubmit(srcIndex, dstIndex);
      waitForGPU();
      render(dstIndex);
      std::swap(srcIndex, dstIndex);
      SDL_Delay(16);
    }
  }

  // Headless benchmark: deterministic seed, time N GPU steps, print one RESULT
  // line. This is the "gpu" rule group (atomic contention => the checksum may
  // vary run to run; the conserved sand count is the deterministic check).
  void runBench(int steps) {
    int srcIndex = 0, dstIndex = 1;
    uint32_t *u32 = reinterpret_cast<uint32_t *>(mappedPtr);
    std::memset(mappedPtr, 0, bufferBytes);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        u32[static_cast<size_t>(y) * width + x] = seedCell(x, y);

    // Record EVERY step into a single command buffer: the src->dst copy
    // (vkCmdCopyBuffer, on the device) and the compute dispatch are separated
    // by pipeline barriers, so the GPU runs all steps back-to-back with no host
    // round-trip and a single fence wait at the end. The original loop copied
    // on the host and fence-waited after every step, which was ~100x slower.
    const VkDeviceSize halfBytes =
        static_cast<VkDeviceSize>(width) * height * sizeof(uint32_t);
    const uint32_t gx = (width + 15) / 16, gy = (height + 15) / 16;

    vkCheck(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer failed");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCheck(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer failed");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    for (int s = 0; s < steps; ++s) {
      VkBufferCopy region{};
      region.srcOffset = static_cast<VkDeviceSize>(srcIndex) * halfBytes;
      region.dstOffset = static_cast<VkDeviceSize>(dstIndex) * halfBytes;
      region.size = halfBytes;
      vkCmdCopyBuffer(commandBuffer, storageBuffer, storageBuffer, 1, &region);
      memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      PushConsts pc{width, height, srcIndex, dstIndex};
      vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(PushConsts), &pc);
      vkCmdDispatch(commandBuffer, gx, gy, 1);
      if (s + 1 < steps)
        memoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
      std::swap(srcIndex, dstIndex); // srcIndex now names the latest half
    }
    // Make the final compute writes visible to the host read below.
    memoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer failed");

    vkCheck(vkResetFences(device, 1, &fence), "vkResetFences failed");
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer;
    auto start = std::chrono::steady_clock::now();
    vkCheck(vkQueueSubmit(computeQueue, 1, &si, fence), "vkQueueSubmit failed");
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    auto end = std::chrono::steady_clock::now();

    size_t cells = static_cast<size_t>(width) * height;
    size_t off = (srcIndex == 0 ? 0 : cells);
    uint64_t c = 14695981039346656037ull, sand = 0;
    for (size_t i = 0; i < cells; ++i) {
      uint32_t v = u32[off + i] & 1u;
      sand += v;
      c = (c ^ static_cast<uint64_t>(v)) * 1099511628211ull;
    }
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double total = static_cast<double>(cells) * steps;
    double mcells = (elapsedMs > 0.0) ? total / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=vulkan rule=gpu width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx sand=%llu\n",
           width, height, steps, elapsedMs, mcells,
           (unsigned long long)c, (unsigned long long)sand);
  }

private:
  // Global pipeline memory barrier recorded into the command buffer.
  void memoryBarrier(VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                     VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = srcAccess;
    mb.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
  }

  // Deterministic per-cell seed (~30% sand), shared with every implementation.
  static uint32_t seedCell(int x, int y) {
    uint32_t h = static_cast<uint32_t>(x) * 374761393u +
                 static_cast<uint32_t>(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (h % 100u) < 30u ? 1u : 0u;
  }

  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;
  std::vector<uint32_t> pixelBuffer;

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t computeQueueFamily = 0;
  VkQueue computeQueue = VK_NULL_HANDLE;

  VkBuffer storageBuffer = VK_NULL_HANDLE;
  VkDeviceMemory storageMemory = VK_NULL_HANDLE;
  void *mappedPtr = nullptr;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkShaderModule computeShaderModule = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  int width;
  int height;
  int renderWidth;
  int renderHeight;
  std::mt19937 rng;
  VkDeviceSize bufferBytes = 0;

  struct PushConsts {
    int32_t width, height, srcIndex, dstIndex;
  };

  void initSDL() {
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Vulkan Sand Simulation (Compute)",
                              SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              renderWidth, renderHeight, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                SDL_TEXTUREACCESS_STREAMING, renderWidth,
                                renderHeight);
    pixelBuffer.resize(renderWidth * renderHeight, 0);
  }
  void initVulkan() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "sandsim_vulkan_compute";
    appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    vkCheck(vkCreateInstance(&ici, nullptr, &instance),
            "vkCreateInstance failed");
    uint32_t physCount = 0;
    vkEnumeratePhysicalDevices(instance, &physCount, nullptr);
    if (!physCount)
      throw std::runtime_error("No Vulkan physical devices found");
    std::vector<VkPhysicalDevice> physList(physCount);
    vkEnumeratePhysicalDevices(instance, &physCount, physList.data());
    phys = physList[0];
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, qProps.data());
    bool found = false;
    for (uint32_t i = 0; i < qCount; ++i) {
      if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeQueueFamily = i;
        found = true;
        break;
      }
    }
    if (!found)
      throw std::runtime_error("No compute queue family found");
    float priority = 1.0f;
    VkDeviceQueueCreateInfo dqci{};
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = computeQueueFamily;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    vkCheck(vkCreateDevice(phys, &dci, nullptr, &device),
            "vkCreateDevice failed");
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
  }
  uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
      if ((typeBits & (1u << i)) &&
          (memProps.memoryTypes[i].propertyFlags & props) == props)
        return i;
    throw std::runtime_error("No suitable memory type");
  }
  void createBuffers() {
    bufferBytes = static_cast<VkDeviceSize>(width) *
                  static_cast<VkDeviceSize>(height) * 2ull * sizeof(uint32_t);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bufferBytes;
    // STORAGE for the compute shader; TRANSFER for the device-side src->dst
    // copy used by the batched benchmark (vkCmdCopyBuffer).
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device, &bci, nullptr, &storageBuffer),
            "vkCreateBuffer failed");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, storageBuffer, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(
        req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkCheck(vkAllocateMemory(device, &mai, nullptr, &storageMemory),
            "vkAllocateMemory failed");
    vkCheck(vkBindBufferMemory(device, storageBuffer, storageMemory, 0),
            "vkBindBufferMemory failed");
    vkCheck(vkMapMemory(device, storageMemory, 0, bufferBytes, 0, &mappedPtr),
            "vkMapMemory failed");
  }
  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 1;
    dslci.pBindings = &binding;
    vkCheck(vkCreateDescriptorSetLayout(device, &dslci, nullptr,
                                        &descriptorSetLayout),
            "vkCreateDescriptorSetLayout failed");
  }
  void createPipeline() {
    auto code = readFile("shaders/sand.comp.spv");
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    vkCheck(vkCreateShaderModule(device, &smci, nullptr, &computeShaderModule),
            "vkCreateShaderModule failed");
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConsts);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descriptorSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout failed");
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = computeShaderModule;
    stage.pName = "main";
    cpci.stage = stage;
    cpci.layout = pipelineLayout;
    vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                     &pipeline),
            "vkCreateComputePipelines failed");
  }
  void createDescriptorPoolAndSet() {
    VkDescriptorPoolSize dps{};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    dpci.maxSets = 1;
    vkCheck(vkCreateDescriptorPool(device, &dpci, nullptr, &descriptorPool),
            "vkCreateDescriptorPool failed");
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &descriptorSetLayout;
    vkCheck(vkAllocateDescriptorSets(device, &dsai, &descriptorSet),
            "vkAllocateDescriptorSets failed");
    VkDescriptorBufferInfo dbi{};
    dbi.buffer = storageBuffer;
    dbi.offset = 0;
    dbi.range = bufferBytes;
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = descriptorSet;
    wds.dstBinding = 0;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.descriptorCount = 1;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
  }
  void createCommandPoolAndBuffer() {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = computeQueueFamily;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCheck(vkCreateCommandPool(device, &cpci, nullptr, &commandPool),
            "vkCreateCommandPool failed");
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkCheck(vkAllocateCommandBuffers(device, &cbai, &commandBuffer),
            "vkAllocateCommandBuffers failed");
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCheck(vkCreateFence(device, &fci, nullptr, &fence),
            "vkCreateFence failed");
  }
  void recordAndSubmit(int srcIndex, int dstIndex) {
    vkCheck(vkResetFences(device, 1, &fence), "vkResetFences failed");
    vkCheck(vkResetCommandBuffer(commandBuffer, 0),
            "vkResetCommandBuffer failed");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCheck(vkBeginCommandBuffer(commandBuffer, &begin),
            "vkBeginCommandBuffer failed");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    PushConsts pc{width, height, srcIndex, dstIndex};
    vkCmdPushConstants(commandBuffer, pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts), &pc);
    uint32_t gx = (width + 15) / 16, gy = (height + 15) / 16;
    vkCmdDispatch(commandBuffer, gx, gy, 1);
    vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer failed");
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer;
    vkCheck(vkQueueSubmit(computeQueue, 1, &si, fence), "vkQueueSubmit failed");
  }
  void waitForGPU() {
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(1'000'000'000));
  }
  void addSandCPU(int bufferIndex, int px, int py, int radius) {
    int x = px / PIXEL_SIZE, y = py / PIXEL_SIZE;
    uint32_t *u32 = reinterpret_cast<uint32_t *>(mappedPtr);
    size_t gridOffset = (bufferIndex == 0 ? 0
                                          : static_cast<size_t>(width) *
                                                static_cast<size_t>(height));
    for (int dy = -radius; dy <= radius; ++dy)
      for (int dx = -radius; dx <= radius; ++dx) {
        int nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
          if (dx * dx + dy * dy <= radius * radius)
            u32[gridOffset +
                static_cast<size_t>(ny) * static_cast<size_t>(width) +
                static_cast<size_t>(nx)] = 1u;
      }
  }
  void randomizeGrid(int bufferIndex, float density = 0.3f) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    uint32_t *u32 = reinterpret_cast<uint32_t *>(mappedPtr);
    size_t gridOffset = (bufferIndex == 0 ? 0
                                          : static_cast<size_t>(width) *
                                                static_cast<size_t>(height));
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        u32[gridOffset + static_cast<size_t>(y) * static_cast<size_t>(width) +
            static_cast<size_t>(x)] = (dist(rng) < density) ? 1u : 0u;
  }
  void copyGrid(int srcIdx, int dstIdx) {
    uint32_t *u32 = reinterpret_cast<uint32_t *>(mappedPtr);
    size_t cells = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t srcOff = (srcIdx == 0 ? 0 : cells);
    size_t dstOff = (dstIdx == 0 ? 0 : cells);
    std::memcpy(u32 + dstOff, u32 + srcOff, cells * sizeof(uint32_t));
  }
  uint32_t readCellCPU(int bufferIndex, int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height)
      return 0u;
    const uint32_t *u32 = reinterpret_cast<const uint32_t *>(mappedPtr);
    size_t gridOffset = (bufferIndex == 0 ? 0
                                          : static_cast<size_t>(width) *
                                                static_cast<size_t>(height));
    return u32[gridOffset +
               static_cast<size_t>(y) * static_cast<size_t>(width) +
               static_cast<size_t>(x)];
  }
  void render(int activeBuffer) {
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        uint32_t sandValue = readCellCPU(activeBuffer, x, y);
        uint32_t color = sandValue ? 0xFFFFFF00u : 0xFF000000u;
        for (int dy = 0; dy < PIXEL_SIZE; ++dy)
          for (int dx = 0; dx < PIXEL_SIZE; ++dx) {
            int rx = x * PIXEL_SIZE + dx, ry = y * PIXEL_SIZE + dy;
            pixelBuffer[static_cast<size_t>(ry) *
                            static_cast<size_t>(renderWidth) +
                        static_cast<size_t>(rx)] = color;
          }
      }
    SDL_UpdateTexture(texture, nullptr, pixelBuffer.data(),
                      renderWidth * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
  }
};

int main(int argc, char **argv) {
  try {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
      int steps = (argc > 2) ? std::atoi(argv[2]) : 1000;
      int width = (argc > 3) ? std::atoi(argv[3]) : 400;
      int height = (argc > 4) ? std::atoi(argv[4]) : 300;
      VulkanSandSim sim(width, height, /*headless=*/true);
      sim.runBench(steps);
    } else {
      VulkanSandSim sim(400, 300);
      sim.run();
    }
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
