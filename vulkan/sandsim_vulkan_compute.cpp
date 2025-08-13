#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>

#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

static void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) { throw std::runtime_error(msg); }
}

static std::vector<char> readFile(const char* path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file");
    size_t size = (size_t)file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

class VulkanSandSim {
public:
    static constexpr int PIXEL_SIZE = 2;
    VulkanSandSim(int w, int h)
        : width(w), height(h), renderWidth(w * PIXEL_SIZE), renderHeight(h * PIXEL_SIZE), rng(std::random_device{}()) {
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
                if (event.type == SDL_QUIT) quit = true;
                else if (event.type == SDL_MOUSEBUTTONDOWN) mouseDown = true;
                else if (event.type == SDL_MOUSEBUTTONUP) mouseDown = false;
                else if (event.type == SDL_MOUSEMOTION) SDL_GetMouseState(&mouseX, &mouseY);
                else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_c) std::memset(mappedPtr, 0, bufferBytes);
                    else if (event.key.keysym.sym == SDLK_r) randomizeGrid(srcIndex);
                }
            }
            if (mouseDown) addSandCPU(srcIndex, mouseX, mouseY, 5);
            copyGrid(srcIndex, dstIndex);
            recordAndSubmit(srcIndex, dstIndex);
            waitForGPU();
            render(dstIndex);
            std::swap(srcIndex, dstIndex);
            SDL_Delay(16);
        }
    }

private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    std::vector<uint32_t> pixelBuffer;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = 0;
    VkQueue computeQueue = VK_NULL_HANDLE;

    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory storageMemory = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
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

    struct PushConsts { int32_t width, height, srcIndex, dstIndex; };

    void initSDL() {
        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("Vulkan Sand Simulation (Compute)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderWidth, renderHeight, 0);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, renderWidth, renderHeight);
        pixelBuffer.resize(renderWidth * renderHeight, 0);
    }
    void initVulkan() {
        VkApplicationInfo appInfo{ }; appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; appInfo.pApplicationName = "sandsim_vulkan_compute"; appInfo.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{ }; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &appInfo; vkCheck(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance failed");
        uint32_t physCount = 0; vkEnumeratePhysicalDevices(instance, &physCount, nullptr); if (!physCount) throw std::runtime_error("No Vulkan physical devices found");
        std::vector<VkPhysicalDevice> physList(physCount); vkEnumeratePhysicalDevices(instance, &physCount, physList.data()); phys = physList[0];
        uint32_t qCount = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, nullptr); std::vector<VkQueueFamilyProperties> qProps(qCount); vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, qProps.data());
        bool found = false; for (uint32_t i = 0; i < qCount; ++i) { if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeQueueFamily = i; found = true; break; } } if (!found) throw std::runtime_error("No compute queue family found");
        float priority = 1.0f; VkDeviceQueueCreateInfo dqci{ }; dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; dqci.queueFamilyIndex = computeQueueFamily; dqci.queueCount = 1; dqci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{ }; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dqci; vkCheck(vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice failed");
        vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
    }
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties memProps; vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i; throw std::runtime_error("No suitable memory type");
    }
    void createBuffers() {
        bufferBytes = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 2ull * sizeof(uint32_t);
        VkBufferCreateInfo bci{ }; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.size = bufferBytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; vkCheck(vkCreateBuffer(device, &bci, nullptr, &storageBuffer), "vkCreateBuffer failed");
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, storageBuffer, &req);
        VkMemoryAllocateInfo mai{ }; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = req.size; mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkCheck(vkAllocateMemory(device, &mai, nullptr, &storageMemory), "vkAllocateMemory failed"); vkCheck(vkBindBufferMemory(device, storageBuffer, storageMemory, 0), "vkBindBufferMemory failed"); vkCheck(vkMapMemory(device, storageMemory, 0, bufferBytes, 0, &mappedPtr), "vkMapMemory failed");
    }
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding binding{ }; binding.binding = 0; binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binding.descriptorCount = 1; binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dslci{ }; dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dslci.bindingCount = 1; dslci.pBindings = &binding; vkCheck(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &descriptorSetLayout), "vkCreateDescriptorSetLayout failed");
    }
    void createPipeline() {
        auto code = readFile("shaders/sand.comp.spv"); VkShaderModuleCreateInfo smci{ }; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; smci.codeSize = code.size(); smci.pCode = reinterpret_cast<const uint32_t*>(code.data()); vkCheck(vkCreateShaderModule(device, &smci, nullptr, &computeShaderModule), "vkCreateShaderModule failed");
        VkPushConstantRange pcr{ }; pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = sizeof(PushConsts);
        VkPipelineLayoutCreateInfo plci{ }; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.setLayoutCount = 1; plci.pSetLayouts = &descriptorSetLayout; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr; vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout), "vkCreatePipelineLayout failed");
        VkComputePipelineCreateInfo cpci{ }; cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; VkPipelineShaderStageCreateInfo stage{ }; stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = computeShaderModule; stage.pName = "main"; cpci.stage = stage; cpci.layout = pipelineLayout; vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline), "vkCreateComputePipelines failed");
    }
    void createDescriptorPoolAndSet() {
        VkDescriptorPoolSize dps{ }; dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dps.descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{ }; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps; dpci.maxSets = 1; vkCheck(vkCreateDescriptorPool(device, &dpci, nullptr, &descriptorPool), "vkCreateDescriptorPool failed");
        VkDescriptorSetAllocateInfo dsai{ }; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool = descriptorPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &descriptorSetLayout; vkCheck(vkAllocateDescriptorSets(device, &dsai, &descriptorSet), "vkAllocateDescriptorSets failed");
        VkDescriptorBufferInfo dbi{ }; dbi.buffer = storageBuffer; dbi.offset = 0; dbi.range = bufferBytes;
        VkWriteDescriptorSet wds{ }; wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds.dstSet = descriptorSet; wds.dstBinding = 0; wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds.descriptorCount = 1; wds.pBufferInfo = &dbi; vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    }
    void createCommandPoolAndBuffer() {
        VkCommandPoolCreateInfo cpci{ }; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci.queueFamilyIndex = computeQueueFamily; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; vkCheck(vkCreateCommandPool(device, &cpci, nullptr, &commandPool), "vkCreateCommandPool failed");
        VkCommandBufferAllocateInfo cbai{ }; cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbai.commandPool = commandPool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1; vkCheck(vkAllocateCommandBuffers(device, &cbai, &commandBuffer), "vkAllocateCommandBuffers failed");
        VkFenceCreateInfo fci{ }; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; vkCheck(vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence failed");
    }
    void recordAndSubmit(int srcIndex, int dstIndex) {
        vkCheck(vkResetFences(device, 1, &fence), "vkResetFences failed"); vkCheck(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer failed");
        VkCommandBufferBeginInfo begin{ }; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; vkCheck(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer failed");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        PushConsts pc{ width, height, srcIndex, dstIndex }; vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts), &pc);
        uint32_t gx = (width + 15) / 16, gy = (height + 15) / 16; vkCmdDispatch(commandBuffer, gx, gy, 1);
        vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer failed");
        VkSubmitInfo si{ }; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &commandBuffer; vkCheck(vkQueueSubmit(computeQueue, 1, &si, fence), "vkQueueSubmit failed");
    }
    void waitForGPU() { vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(1'000'000'000)); }
    void addSandCPU(int bufferIndex, int px, int py, int radius) {
        int x = px / PIXEL_SIZE, y = py / PIXEL_SIZE; uint32_t* u32 = reinterpret_cast<uint32_t*>(mappedPtr);
        size_t gridOffset = (bufferIndex == 0 ? 0 : static_cast<size_t>(width) * static_cast<size_t>(height));
        for (int dy = -radius; dy <= radius; ++dy) for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx, ny = y + dy; if (nx >= 0 && nx < width && ny >= 0 && ny < height) if (dx*dx + dy*dy <= radius*radius)
                u32[gridOffset + static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx)] = 1u; }
    }
    void randomizeGrid(int bufferIndex, float density = 0.3f) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f); uint32_t* u32 = reinterpret_cast<uint32_t*>(mappedPtr);
        size_t gridOffset = (bufferIndex == 0 ? 0 : static_cast<size_t>(width) * static_cast<size_t>(height));
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
            u32[gridOffset + static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = (dist(rng) < density) ? 1u : 0u;
    }
    void copyGrid(int srcIdx, int dstIdx) {
        uint32_t* u32 = reinterpret_cast<uint32_t*>(mappedPtr); size_t cells = static_cast<size_t>(width) * static_cast<size_t>(height);
        size_t srcOff = (srcIdx == 0 ? 0 : cells); size_t dstOff = (dstIdx == 0 ? 0 : cells); std::memcpy(u32 + dstOff, u32 + srcOff, cells * sizeof(uint32_t));
    }
    uint32_t readCellCPU(int bufferIndex, int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0u; const uint32_t* u32 = reinterpret_cast<const uint32_t*>(mappedPtr);
        size_t gridOffset = (bufferIndex == 0 ? 0 : static_cast<size_t>(width) * static_cast<size_t>(height));
        return u32[gridOffset + static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }
    void render(int activeBuffer) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            uint32_t sandValue = readCellCPU(activeBuffer, x, y); uint32_t color = sandValue ? 0xFFFFFF00u : 0xFF000000u;
            for (int dy = 0; dy < PIXEL_SIZE; ++dy) for (int dx = 0; dx < PIXEL_SIZE; ++dx) {
                int rx = x * PIXEL_SIZE + dx, ry = y * PIXEL_SIZE + dy; pixelBuffer[static_cast<size_t>(ry) * static_cast<size_t>(renderWidth) + static_cast<size_t>(rx)] = color; } }
        SDL_UpdateTexture(texture, nullptr, pixelBuffer.data(), renderWidth * sizeof(uint32_t)); SDL_RenderClear(renderer); SDL_RenderCopy(renderer, texture, nullptr, nullptr); SDL_RenderPresent(renderer);
    }
};

int main() { try { VulkanSandSim sim(400, 300); sim.run(); } catch (const std::exception& ex) { std::cerr << "Error: " << ex.what() << std::endl; return 1; } return 0; }


