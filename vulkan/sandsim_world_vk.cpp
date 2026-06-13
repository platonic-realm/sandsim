/*
 * sandsim - multi-material streaming world, Vulkan compute (bit-identical to CPU)
 *
 * The same order-independent rule as cpp/simd_core.h, on the GPU. Each frame is
 * 16 compute dispatches (one per disjoint sub-pass) with a memory barrier
 * between (shaders/world.comp). A thread decides whether its cell is the SOURCE
 * of a swap purely from its (x,y) coordinates, so the in-place update is
 * race-free and reproduces the CPU result exactly.
 *
 * The live GW x GH window is a host-visible (mapped) storage buffer, so the CPU
 * streaming reads/writes it directly -- the chunk<->disk logic is identical to
 * the C++ build, no staging copies. Same seed + same camera path => the
 * whole-world checksum matches the C++ and OpenGL builds bit-for-bit.
 *
 * Modes:
 *   --bench [steps] [wch] [hch]   headless streaming benchmark (one RESULT line)
 *   (default)                     interactive viewer (arrows pan, number keys paint)
 */

#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include "../ui.h"       // on-screen material palette

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, OIL = 5, FIRE = 6, LAVA = 7, STEAM = 8, WOOD = 9, PLANT = 10, ACID = 11, SMOKE = 12, GLASS = 13, ICE = 14, MATERIAL_COUNT = 15 };
enum { SG_DOWN, SG_GAS, SG_HORIZ };

static constexpr int CHUNK = 64;
static constexpr int PAD = 16;

static const uint32_t kColors[MATERIAL_COUNT] = {
    0xFF000000u, 0xFF808080u, 0xFFE2C878u, 0xFF4488FFu, 0xFFB0C4DEu, 0xFF8E44ADu, 0xFFFF5A1Eu, 0xFFCF1B0Bu, 0xFFDCE4ECu, 0xFF8B5A2Bu, 0xFF3AA84Au, 0xFFB8F000u, 0xFF585860u, 0xFFAEE0E8u, 0xFFCDEBFFu,
};

// Window resolution + virtual-pixel scale + simulation rate. simHz is steps/second,
// decoupled from the render rate so the physics runs at the same wall-clock speed
// on every backend.
struct ViewCfg { int winW = 1024, winH = 768, scale = 2, simHz = 60; };
static ViewCfg parseView(int argc, char* argv[]) {
    ViewCfg c;
    if (const char* e = getenv("SANDSIM_RES"))   std::sscanf(e, "%dx%d", &c.winW, &c.winH);
    if (const char* e = getenv("SANDSIM_SCALE")) c.scale = std::atoi(e);
    if (const char* e = getenv("SANDSIM_SPS"))   c.simHz = std::atoi(e);
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--res") && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &c.winW, &c.winH);
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) c.scale = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--sps") && i + 1 < argc) c.simHz = std::atoi(argv[++i]);
    }
    if (c.scale < 1) c.scale = 1;
    if (c.simHz < 1) c.simHz = 1;
    if (c.winW < CHUNK * c.scale) c.winW = CHUNK * c.scale;
    if (c.winH < CHUNK * c.scale) c.winH = CHUNK * c.scale;
    return c;
}

static inline uint32_t hashCoord(int gx, int gy) {
    uint32_t h = (uint32_t)gx * 374761393u + (uint32_t)gy * 668265263u;
    return (h ^ (h >> 13)) * 1274126177u;
}
static inline uint8_t seedMat(int gx, int gy) {
    if (gy % 40 == 39 && (gx % 11 != 0)) return WALL;
    uint32_t r = hashCoord(gx, gy) % 100u;
    switch ((gy / 40) % 3) {
        case 0:  return (r < 35u) ? SAND  : EMPTY;
        case 1:  return (r < 30u) ? WATER : EMPTY;
        default: return (r < 18u) ? GAS   : EMPTY;
    }
}

struct Pass { int32_t type, dx, dy, parity, grp; };
static const Pass kPasses[16] = {
    {0,  0,  1, 0, SG_DOWN}, {0,  0,  1, 1, SG_DOWN},
    {1, -1,  1, 0, SG_DOWN}, {1, -1,  1, 1, SG_DOWN},
    {1,  1,  1, 0, SG_DOWN}, {1,  1,  1, 1, SG_DOWN},
    {0,  0, -1, 0, SG_GAS},  {0,  0, -1, 1, SG_GAS},
    {1, -1, -1, 0, SG_GAS},  {1, -1, -1, 1, SG_GAS},
    {1,  1, -1, 0, SG_GAS},  {1,  1, -1, 1, SG_GAS},
    {2, -1,  0, 0, SG_HORIZ},{2, -1,  0, 1, SG_HORIZ},
    {2,  1,  0, 0, SG_HORIZ},{2,  1,  0, 1, SG_HORIZ},
};

struct PushConsts { int32_t SW, X0, X1, Y0, Y1, type, dx, dy, parity, grp, frame; };

static void vkCheck(VkResult r, const char* msg) { if (r != VK_SUCCESS) throw std::runtime_error(msg); }

// Look for the SPIR-V next to the executable, then in the current directory, so
// the binary runs from anywhere (e.g. the repo root via tools/benchmark.sh).
static std::string shaderPath() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::filesystem::path p = std::filesystem::path(buf).parent_path() / "shaders" / "world.comp.spv";
        if (std::filesystem::exists(p)) return p.string();
    }
    return "shaders/world.comp.spv";
}
static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    size_t n = (size_t)f.tellg(); std::vector<char> b(n); f.seekg(0); f.read(b.data(), n); return b;
}

// --------------------------------------------------------------------------- world
class VkWorld {
public:
    VkWorld(int gw, int gh, int wbox, int hbox, std::string dir)
        : gw(gw), gh(gh), LW(gw * CHUNK), LH(gh * CHUNK), SW(LW + 2 * PAD), SH(LH + 2 * PAD),
          X0(PAD), X1(PAD + LW), Y0(PAD), Y1(PAD + LH),
          gridBytes((VkDeviceSize)SW * SH * sizeof(uint32_t)),
          wbox(wbox), hbox(hbox), dir(std::move(dir)) {
        std::filesystem::create_directories(this->dir);
        initVulkan();
        createBuffers();
        createPipeline();
        createDescriptor();
        createCommands();
        for (size_t i = 0; i < (size_t)SW * SH; ++i) stagingPtr[i] = WALL;  // border stays WALL
        uploadCells();   // seed the device-local grid (including its WALL border)
    }
    ~VkWorld() {
        vkDeviceWaitIdle(device);
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyDescriptorPool(device, descPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        vkDestroyShaderModule(device, shaderMod, nullptr);
        vkDestroyBuffer(device, cellsBuf, nullptr);   vkFreeMemory(device, cellsMem, nullptr);
        vkDestroyBuffer(device, movedBuf, nullptr);   vkFreeMemory(device, movedMem, nullptr);
        vkDestroyBuffer(device, stagingBuf, nullptr); vkFreeMemory(device, stagingMem, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    void generateAllToDisk() {
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) { genBox(cx, cy, buf); writeBox(cx, cy, buf); }
    }

    void setWindow(int camCx, int camCy) {
        if (windowValid && camCx == winCx && camCy == winCy) return;
        syncDown();                                // pull the GPU's latest into staging
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        if (windowValid)
            for (int y = 0; y < gh; ++y)
                for (int x = 0; x < gw; ++x) { extractChunk(x, y, buf); writeBox(winCx + x, winCy + y, buf); }
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                if (!readBox(camCx + x, camCy + y, buf)) genBox(camCx + x, camCy + y, buf);
                injectChunk(x, y, buf);
            }
        winCx = camCx; winCy = camCy; windowValid = true;
        residentMax = gw * gh;
        syncUp();                                  // push the new window to the GPU
    }

    void step() { stepN(1); }

    // Advance n frames. Consecutive frames (between camera moves) are recorded
    // into one command buffer and submitted together, so the GPU runs them
    // back-to-back with a single fence wait instead of one per frame. Capped per
    // submit so a long run can't trip the GPU watchdog.
    void stepN(int n) {
        const int kBatch = 128;
        while (n > 0) {
            int b = (n < kBatch) ? n : kBatch;
            beginCmd(); recordFrames(b); endSubmitWait();
            frame += (uint32_t)b;
            n -= b;
        }
        gpuAhead = true;                           // device grid is now ahead of staging
    }

    // Make the live window available to the host (for rendering after a step).
    void present() { syncDown(); }

    // Interactive helper: advance one frame AND copy the result back to the host
    // staging buffer in a single command submission (one fence wait, not two).
    void stepAndPresent() {
        beginCmd();
        recordFrames(1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy region{0, 0, gridBytes};
        vkCmdCopyBuffer(cmd, cellsBuf, stagingBuf, 1, &region);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        endSubmitWait();
        frame += 1;
        gpuAhead = false;          // staging now matches the device
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        if (material == FIRE || material == LAVA || material == STEAM || material == PLANT || material == ACID || material == SMOKE) hasReactive = true;
        syncDown();
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < LW && ny >= 0 && ny < LH && dx * dx + dy * dy <= radius * radius)
                    stagingPtr[(size_t)(ny + Y0) * SW + (nx + X0)] = material;
            }
        syncUp();
    }
    uint8_t viewCell(int lx, int ly) const { return (uint8_t)(stagingPtr[(size_t)(ly + Y0) * SW + (lx + X0)] & 0xFFu); }

    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) {
        syncDown();
        for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        uint64_t c = 14695981039346656037ull;
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) {
                bool inWin = windowValid && cx >= winCx && cx < winCx + gw && cy >= winCy && cy < winCy + gh;
                const uint8_t* data;
                if (inWin) { extractChunk(cx - winCx, cy - winCy, buf); data = buf.data(); }
                else { readBox(cx, cy, buf); data = buf.data(); }
                for (int i = 0; i < CHUNK * CHUNK; ++i) { uint8_t v = data[i]; counts[v]++; c = (c ^ v) * 1099511628211ull; }
            }
        checksum = c;
    }

    int residentMaxCount() const { return residentMax; }
    long long diskWrites() const { return nWrites; }
    long long diskReads() const { return nReads; }
    int winChunksW() const { return gw; }
    int winChunksH() const { return gh; }
    int cellsW() const { return LW; }
    int cellsH() const { return LH; }

private:
    const int gw, gh;                 // live window in chunks
    const int LW, LH;                 // live window in cells
    const int SW, SH;                 // padded stride / height
    const int X0, X1, Y0, Y1;         // interior cell range
    VkDeviceSize gridBytes;
    int wbox, hbox;
    std::string dir;
    int winCx = 0, winCy = 0;
    bool windowValid = false;
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;
    uint32_t frame = 0;
    bool hasReactive = false;        // gates the reaction dispatches (fire/lava)

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t qFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkBuffer cellsBuf = VK_NULL_HANDLE, movedBuf = VK_NULL_HANDLE, stagingBuf = VK_NULL_HANDLE;
    VkDeviceMemory cellsMem = VK_NULL_HANDLE, movedMem = VK_NULL_HANDLE, stagingMem = VK_NULL_HANDLE;
    uint32_t* stagingPtr = nullptr;      // mapped host staging shadow of the live grid
    bool gpuAhead = false;               // device cellsBuf holds newer data than staging
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shaderMod = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
        throw std::runtime_error("no suitable memory type");
    }
    void initVulkan() {
        VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "sandsim_world_vk"; app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
        vkCheck(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
        uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
        if (!n) throw std::runtime_error("no Vulkan device");
        std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(instance, &n, devs.data()); phys = devs[0];
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qp.data());
        bool found = false;
        for (uint32_t i = 0; i < qn; ++i) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qFamily = i; found = true; break; }
        if (!found) throw std::runtime_error("no compute queue");
        float pr = 1.0f;
        VkDeviceQueueCreateInfo dq{}; dq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dq.queueFamilyIndex = qFamily; dq.queueCount = 1; dq.pQueuePriorities = &pr;
        VkDeviceCreateInfo dci{}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dq;
        vkCheck(vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, qFamily, 0, &queue);
    }
    void makeBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                    VkBuffer& buf, VkDeviceMemory& mem, void** map) {
        VkBufferCreateInfo bci{}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCheck(vkCreateBuffer(device, &bci, nullptr, &buf), "vkCreateBuffer");
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, buf, &req);
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
        vkCheck(vkAllocateMemory(device, &mai, nullptr, &mem), "vkAllocateMemory");
        vkCheck(vkBindBufferMemory(device, buf, mem, 0), "vkBindBufferMemory");
        if (map) vkCheck(vkMapMemory(device, mem, 0, bytes, 0, map), "vkMapMemory");
    }
    void createBuffers() {
        // cells + moved live in fast DEVICE_LOCAL VRAM (the compute hot path);
        // a HOST_VISIBLE staging buffer is what the CPU streams into, copied to/
        // from the device only at camera moves / seed / summary.
        makeBuffer(gridBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, cellsBuf, cellsMem, nullptr);
        makeBuffer(gridBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, movedBuf, movedMem, nullptr);
        makeBuffer(gridBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuf, stagingMem, (void**)&stagingPtr);
    }
    void createPipeline() {
        auto code = readFile(shaderPath());
        VkShaderModuleCreateInfo smci{}; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = code.size(); smci.pCode = (const uint32_t*)code.data();
        vkCheck(vkCreateShaderModule(device, &smci, nullptr, &shaderMod), "vkCreateShaderModule");
        VkDescriptorSetLayoutBinding b[2]{};
        for (int i = 0; i < 2; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo dl{}; dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dl.bindingCount = 2; dl.pBindings = b;
        vkCheck(vkCreateDescriptorSetLayout(device, &dl, nullptr, &descLayout), "descSetLayout");
        VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.size = sizeof(PushConsts);
        VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &descLayout; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
        vkCheck(vkCreatePipelineLayout(device, &pl, nullptr, &pipeLayout), "pipeLayout");
        VkPipelineShaderStageCreateInfo st{}; st.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st.stage = VK_SHADER_STAGE_COMPUTE_BIT; st.module = shaderMod; st.pName = "main";
        VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.stage = st; cp.layout = pipeLayout;
        vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline), "computePipeline");
    }
    void createDescriptor() {
        VkDescriptorPoolSize ps{}; ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.poolSizeCount = 1; dp.pPoolSizes = &ps; dp.maxSets = 1;
        vkCheck(vkCreateDescriptorPool(device, &dp, nullptr, &descPool), "descPool");
        VkDescriptorSetAllocateInfo da{}; da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        da.descriptorPool = descPool; da.descriptorSetCount = 1; da.pSetLayouts = &descLayout;
        vkCheck(vkAllocateDescriptorSets(device, &da, &descSet), "descSet");
        VkDescriptorBufferInfo bi[2]{};
        bi[0].buffer = cellsBuf; bi[0].range = gridBytes;
        bi[1].buffer = movedBuf; bi[1].range = gridBytes;
        VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = descSet;
            w[i].dstBinding = i; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].descriptorCount = 1; w[i].pBufferInfo = &bi[i]; }
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }
    void createCommands() {
        VkCommandPoolCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex = qFamily; cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCheck(vkCreateCommandPool(device, &cp, nullptr, &cmdPool), "cmdPool");
        VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = cmdPool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        vkCheck(vkAllocateCommandBuffers(device, &ca, &cmd), "cmdBuf");
        VkFenceCreateInfo fc{}; fc.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCheck(vkCreateFence(device, &fc, nullptr, &fence), "fence");
    }
    void barrier(VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
        VkMemoryBarrier mb{}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = src; mb.dstAccessMask = dst;
        vkCmdPipelineBarrier(cmd, ss, ds, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void beginCmd() {
        vkCheck(vkResetCommandBuffer(cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo bg{}; bg.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkCheck(vkBeginCommandBuffer(cmd, &bg), "beginCmd");
    }
    void endSubmitWait() {
        vkCheck(vkEndCommandBuffer(cmd), "endCmd");
        vkCheck(vkResetFences(device, 1, &fence), "vkResetFences");
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkCheck(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit");
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    // Record n frames (operating on the DEVICE-LOCAL cells/moved buffers). Each
    // frame: clear moved, then the 16 passes (each followed by a barrier). The
    // leading barrier makes a prior upload copy / previous batch visible.
    void recordFrames(int n) {
        barrier(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
        for (int f = 0; f < n; ++f) {
            vkCmdFillBuffer(cmd, movedBuf, 0, gridBytes, 0);
            barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            for (int i = 0; i < 16; ++i) {
                const Pass& p = kPasses[i];
                PushConsts pc{SW, X0, X1, Y0, Y1, p.type, p.dx, p.dy, p.parity, p.grp, 0};
                vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, LW / 16, LH / 16, 1);
                barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }
            if (hasReactive) {                      // reactions (gated): see shader pass types
                int decay = (int)(frame + (uint32_t)f);
                for (int t : {3,4,5,6,7,8,9,10,11,12,13,14,15}) {  // + glass, ice melt
                    PushConsts pc{SW, X0, X1, Y0, Y1, t, 0, 0, 0, 0, decay};
                    vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, LW / 16, LH / 16, 1);
                    barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                }
            }
            // next frame's moved-clear (TRANSFER) must wait for this frame's shader writes
            if (f + 1 < n)
                barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        }
    }

    // device cellsBuf -> host stagingBuf (so the CPU can read/stream the window).
    void downloadCells() {
        beginCmd();
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy region{0, 0, gridBytes};
        vkCmdCopyBuffer(cmd, cellsBuf, stagingBuf, 1, &region);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        endSubmitWait();
    }
    // host stagingBuf -> device cellsBuf (push the seeded / streamed window).
    void uploadCells() {
        beginCmd();
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy region{0, 0, gridBytes};
        vkCmdCopyBuffer(cmd, stagingBuf, cellsBuf, 1, &region);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        endSubmitWait();
    }
    void syncDown() { if (gpuAhead) { downloadCells(); gpuAhead = false; } }
    void syncUp()   { uploadCells(); gpuAhead = false; }

    void extractChunk(int gx, int gy, std::vector<uint8_t>& out) const {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                out[ly * CHUNK + lx] = (uint8_t)(stagingPtr[(size_t)(Y0 + gy * CHUNK + ly) * SW + (X0 + gx * CHUNK + lx)] & 0xFFu);
    }
    void injectChunk(int gx, int gy, const std::vector<uint8_t>& in) {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx) {
                uint8_t v = in[ly * CHUNK + lx];
                if (v == FIRE || v == LAVA || v == STEAM || v == PLANT || v == ACID || v == SMOKE) hasReactive = true;
                stagingPtr[(size_t)(Y0 + gy * CHUNK + ly) * SW + (X0 + gx * CHUNK + lx)] = v;
            }
    }
    void genBox(int cx, int cy, std::vector<uint8_t>& buf) {
        for (int y = 0; y < CHUNK; ++y)
            for (int x = 0; x < CHUNK; ++x)
                buf[y * CHUNK + x] = seedMat(cx * CHUNK + x, cy * CHUNK + y);
    }
    std::string path(int cx, int cy) const {
        char n[64]; std::snprintf(n, sizeof(n), "/b_%d_%d.bin", cx, cy); return dir + n;
    }
    void writeBox(int cx, int cy, const std::vector<uint8_t>& buf) {
        std::ofstream f(path(cx, cy), std::ios::binary);
        f.write((const char*)buf.data(), CHUNK * CHUNK); ++nWrites;
    }
    bool readBox(int cx, int cy, std::vector<uint8_t>& buf) {
        std::ifstream f(path(cx, cy), std::ios::binary);
        if (!f) return false;
        f.read((char*)buf.data(), CHUNK * CHUNK); ++nReads;
        return true;
    }
};

// --------------------------------------------------------------------------- modes
static int runBench(int steps, int wbox, int hbox) {
    const int gw = 4, gh = 4;   // fixed live window for the bit-identical reference
    if (wbox < gw) wbox = gw;
    if (hbox < gh) hbox = gh;
    std::string dir = "/tmp/sandsim_world_vk_" + std::to_string(steps) + "_" +
                      std::to_string(wbox) + "x" + std::to_string(hbox);
    std::filesystem::remove_all(dir);
    VkWorld world(gw, gh, wbox, hbox, dir);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int nposX = wbox - gw + 1, nposY = hbox - gh + 1, nWin = nposX * nposY;
    (void)nposY;
    auto camFor = [&](int s, int& cx, int& cy) {
        int visit = (int)((long long)s * nWin / steps);
        if (visit >= nWin) visit = nWin - 1;
        int row = visit / nposX, col = visit % nposX;
        cx = (row % 2 == 0) ? col : (nposX - 1 - col); cy = row;
    };
    auto start = std::chrono::steady_clock::now();
    int s = 0;
    while (s < steps) {                         // batch consecutive same-window frames
        int cx, cy; camFor(s, cx, cy);
        world.setWindow(cx, cy);
        int run = 1, ncx, ncy;
        while (s + run < steps) { camFor(s + run, ncx, ncy); if (ncx != cx || ncy != cy) break; ++run; }
        world.stepN(run);
        s += run;
    }
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    world.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double mc = (ms > 0.0) ? (double)world.cellsW() * world.cellsH() * steps / (ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=vulkan rule=world window=%dx%d wbox=%d hbox=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           gw, gh, wbox, hbox, steps, ms, mc, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(), conserved ? "yes" : "no");
    std::filesystem::remove_all(dir);
    return conserved ? 0 : 2;
}

static int runInteractive(ViewCfg cfg) {
    const int PIXEL = cfg.scale;
    const int gw = std::max(1, (cfg.winW / PIXEL) / CHUNK);
    const int gh = std::max(1, (cfg.winH / PIXEL) / CHUNK);
    const int WBOX = 2 * gw, HBOX = 2 * gh;
    const int LW = gw * CHUNK, LH = gh * CHUNK;
    const int renderW = LW * PIXEL, renderH = LH * PIXEL;
    std::string dir = "/tmp/sandsim_world_vk_interactive";
    std::filesystem::remove_all(dir);
    VkWorld world(gw, gh, WBOX, HBOX, dir);
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.setWindow(camCx, camCy);
    uint8_t current = SAND;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* win = SDL_CreateWindow(
        "Vulkan World - arrows pan, click palette to pick, [ ] brush",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, renderW, renderH, 0);
    // The compute step fence-waits to completion before we present, so the
    // accelerated (GPU) blit no longer overlaps the Vulkan compute -- it's the
    // snappy default. Set SANDSIM_VK_RENDERER=software to force a CPU blit if the
    // GPU-shared path ever flickers on your driver.
    const char* rk = getenv("SANDSIM_VK_RENDERER");
    bool wantSoft = (rk && std::strcmp(rk, "software") == 0);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, wantSoft ? SDL_RENDERER_SOFTWARE : SDL_RENDERER_ACCELERATED);
    if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    SDL_RenderSetLogicalSize(ren, renderW, renderH);        // scales content to the actual output
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, renderW, renderH);
    SDL_RendererInfo ri{}; SDL_GetRendererInfo(ren, &ri);
    int outW = renderW, outH = renderH; SDL_GetRendererOutputSize(ren, &outW, &outH);
    fprintf(stderr, "sandsim [vulkan]: window %dx%d (output %dx%d), renderer=%s, scale %d, "
            "grid %dx%d chunks = %dx%d cells, %d steps/s\n",
            renderW, renderH, outW, outH, ri.name, PIXEL, gw, gh, LW, LH, cfg.simHz);

    static const uint8_t kSwatch[15] = {EMPTY, WALL, SAND, WATER, GAS, OIL, FIRE, LAVA, STEAM, WOOD, PLANT, ACID, SMOKE, GLASS, ICE};
    uint32_t swatchCol[15];
    for (int i = 0; i < 15; ++i) swatchCol[i] = kColors[kSwatch[i]];
    ui::Palette pal = ui::palette(renderW, 15);
    int brushRadius = 4;
    bool painting = false;
    auto selectedIdx = [&]() { for (int i = 0; i < 15; ++i) if (kSwatch[i] == current) return i; return -1; };

    std::vector<uint32_t> pixels((size_t)renderW * renderH, 0);
    bool quit = false; int mouseX = 0, mouseY = 0; SDL_Event e;
    const double stepDt = 1.0 / cfg.simHz;          // seconds per simulation step
    double acc = 0.0;
    auto last = std::chrono::steady_clock::now();
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            else if (e.type == SDL_MOUSEBUTTONDOWN) {
                float flx, fly; SDL_RenderWindowToLogical(ren, e.button.x, e.button.y, &flx, &fly);
                int h = ui::hit(pal, (int)flx, (int)fly);
                if (h >= 0) current = kSwatch[h];     // clicked a palette swatch
                else { painting = true; world.paint((int)flx / PIXEL, (int)fly / PIXEL, current, brushRadius); }
            }
            else if (e.type == SDL_MOUSEBUTTONUP) painting = false;
            else if (e.type == SDL_MOUSEMOTION) SDL_GetMouseState(&mouseX, &mouseY);
            else if (e.type == SDL_KEYDOWN) switch (e.key.keysym.sym) {
                case SDLK_0: current = EMPTY; break; case SDLK_1: current = WALL; break;
                case SDLK_2: current = SAND;  break; case SDLK_3: current = WATER; break;
                case SDLK_4: current = GAS;   break;
                case SDLK_5: current = OIL;   break;
                case SDLK_6: current = FIRE;  break;
                case SDLK_7: current = LAVA;  break;
                case SDLK_8: current = STEAM; break;
                case SDLK_9: current = WOOD;  break;
                case SDLK_p: current = PLANT; break;
                case SDLK_a: current = ACID;  break;
                case SDLK_m: current = SMOKE; break;
                case SDLK_g: current = GLASS; break;
                case SDLK_i: current = ICE; break;
                case SDLK_LEFTBRACKET:  if (brushRadius > 0)  brushRadius--; break;
                case SDLK_RIGHTBRACKET: if (brushRadius < 32) brushRadius++; break;
                case SDLK_LEFT:  if (camCx > 0) camCx--; break;
                case SDLK_RIGHT: if (camCx < WBOX - gw) camCx++; break;
                case SDLK_UP:    if (camCy > 0) camCy--; break;
                case SDLK_DOWN:  if (camCy < HBOX - gh) camCy++; break;
            }
        }
        world.setWindow(camCx, camCy);
        if (painting) {
            float flx, fly; SDL_RenderWindowToLogical(ren, mouseX, mouseY, &flx, &fly);
            world.paint((int)flx / PIXEL, (int)fly / PIXEL, current, brushRadius);
        }
        // Advance the simulation by however much real time elapsed, so physics
        // runs at cfg.simHz steps/s regardless of the render frame rate.
        auto nowT = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double>(nowT - last).count();
        last = nowT;
        int ran = 0;
        while (acc >= stepDt && ran < 8) { acc -= stepDt; ++ran; }
        if (acc > stepDt) acc = stepDt;             // drop backlog after a stall
        if (ran > 1) world.stepN(ran - 1);
        if (ran > 0) world.stepAndPresent();        // last step + readback in one submit
        static int tick = 0; ++tick;                // render clock for the flame/lava flicker
        for (int y = 0; y < LH; ++y)
            for (int x = 0; x < LW; ++x) {
                uint8_t m = world.viewCell(x, y);
                uint32_t color = kColors[m];
                if (m == FIRE || m == LAVA) color = ui::flicker(color, x, y, tick);
                for (int dy = 0; dy < PIXEL; ++dy)
                    for (int dx = 0; dx < PIXEL; ++dx)
                        pixels[(size_t)(y * PIXEL + dy) * renderW + (x * PIXEL + dx)] = color;
            }
        ui::draw(pixels.data(), renderW, renderH, pal, swatchCol, selectedIdx());
        SDL_UpdateTexture(tex, nullptr, pixels.data(), renderW * (int)sizeof(uint32_t));
        SDL_RenderClear(ren); SDL_RenderCopy(ren, tex, nullptr, nullptr); SDL_RenderPresent(ren);
        SDL_Delay(16);
    }
    SDL_DestroyTexture(tex); SDL_DestroyRenderer(ren); SDL_DestroyWindow(win); SDL_Quit();
    std::filesystem::remove_all(dir);
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            int steps = (argc > 2) ? std::atoi(argv[2]) : 600;
            int wbox  = (argc > 3) ? std::atoi(argv[3]) : 6;
            int hbox  = (argc > 4) ? std::atoi(argv[4]) : 6;
            return runBench(steps, wbox, hbox);
        }
        return runInteractive(parseView(argc, argv));
    } catch (const std::exception& ex) {
        fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
