// sandsim - OpenGL implementation
//
// GPU falling-sand using an OpenGL 4.3 compute shader. It follows the same
// model as the Vulkan version (vulkan/shaders/sand.comp): a single SSBO holds
// the grid twice (a "src" half and a "dst" half). Each step the host copies
// src->dst, then the compute shader atomically claims destination cells and
// clears the source position, so unmoved particles stay put and no two
// particles land on the same cell. This is the "gpu" rule group: the parity
// based left/right preference and atomic contention make it distinct from the
// scalar CPU rule, so its checksum is its own (and, because contention is
// resolved by GPU scheduling order, may vary slightly run to run; the sand
// count is conserved and printed for a deterministic sanity check).
//
// Modes:
//   (default)                 GLFW window, 400x300 grid rendered 2x (800x600).
//   --bench [steps] [w] [h]   headless (hidden context): time N compute steps
//                             and print one RESULT line. Requires a reachable
//                             GPU/display for context creation.
//
// Shaders are embedded as raw strings below so the program is independent of
// the working directory.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

static constexpr int PIXEL_SIZE = 2;

// ---------------------------------------------------------------------------
// Shared helpers (seed + checksum identical to every other implementation).
// ---------------------------------------------------------------------------
static inline uint8_t seedCell(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (h % 100u) < 30u ? 1u : 0u;
}
static uint64_t checksum(const std::vector<uint32_t>& grid) {
    uint64_t c = 14695981039346656037ull;
    for (uint32_t cell : grid) c = (c ^ (uint64_t)(cell & 1u)) * 1099511628211ull;
    return c;
}

// ---------------------------------------------------------------------------
// Embedded GLSL.
// ---------------------------------------------------------------------------
static const char* kComputeSrc = R"GLSL(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Grid { uint cells[]; };
uniform int uWidth;
uniform int uHeight;
uniform int uSrc;
uniform int uDst;
int gstride() { return uWidth * uHeight; }
int idx(int x, int y) { return y * uWidth + x; }
bool inB(int x, int y) { return x >= 0 && x < uWidth && y >= 0 && y < uHeight; }
uint readCell(int x, int y, int halfIdx) {
    if (!inB(x, y)) return 1u;
    return cells[halfIdx * gstride() + idx(x, y)];
}
bool claimDst(int x, int y) {
    return atomicCompSwap(cells[uDst * gstride() + idx(x, y)], 0u, 1u) == 0u;
}
void clearDst(int x, int y) { cells[uDst * gstride() + idx(x, y)] = 0u; }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= uWidth || y >= uHeight) return;
    if (readCell(x, y, uSrc) != 1u) return;
    if (readCell(x, y + 1, uSrc) == 0u && claimDst(x, y + 1)) { clearDst(x, y); return; }
    bool preferLeft = (((x + y) & 1) == 0);
    int dx1 = preferLeft ? -1 : 1;
    int dx2 = -dx1;
    if (readCell(x + dx1, y + 1, uSrc) == 0u && claimDst(x + dx1, y + 1)) { clearDst(x, y); return; }
    if (readCell(x + dx2, y + 1, uSrc) == 0u && claimDst(x + dx2, y + 1)) { clearDst(x, y); return; }
}
)GLSL";

static const char* kPresentVert = R"GLSL(
#version 430
void main() {
    vec2 v[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)GLSL";

static const char* kPresentFrag = R"GLSL(
#version 430
layout(std430, binding = 0) buffer Grid { uint cells[]; };
uniform int uWidth;
uniform int uHeight;
uniform int uCur;
uniform int uPixel;
out vec4 fragColor;
void main() {
    int gx = int(gl_FragCoord.x) / uPixel;
    int gy = uHeight - 1 - (int(gl_FragCoord.y) / uPixel);
    uint v = 0u;
    if (gx >= 0 && gx < uWidth && gy >= 0 && gy < uHeight)
        v = cells[uCur * uWidth * uHeight + gy * uWidth + gx];
    fragColor = (v != 0u) ? vec4(1.0, 1.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}
)GLSL";

// ---------------------------------------------------------------------------
// GL helpers.
// ---------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "shader compile error:\n%s\n", log);
        exit(1);
    }
    return s;
}
static GLuint linkProgram(std::vector<GLuint> shaders) {
    GLuint p = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "program link error:\n%s\n", log);
        exit(1);
    }
    for (GLuint s : shaders) glDeleteShader(s);
    return p;
}

// Create a GL 4.3 core context (hidden when !visible) and load entry points.
// GLEW resolves symbols through GLX, so prefer the X11 backend (works under
// XWayland too) and fall back to whatever GLFW auto-detects.
static GLFWwindow* initGL(bool visible, int w, int h, const char* title) {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    if (!glfwInit()) {
        glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
        if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return nullptr; }
    }
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if (!win) { fprintf(stderr, "no GL 4.3 context (need a reachable GPU/display)\n"); glfwTerminate(); return nullptr; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    GLenum ge = glewInit();
    // NO_GLX_DISPLAY is benign on EGL/Wayland-backed contexts; the entry points
    // still load. Anything else is fatal.
    if (ge != GLEW_OK && ge != GLEW_ERROR_NO_GLX_DISPLAY) {
        fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(ge));
        glfwTerminate();
        return nullptr;
    }
    glGetError(); // discard the benign error glewInit may raise on core profile
    return win;
}

// Create the grid SSBO with the src half seeded; returns the buffer name.
static GLuint makeGridBuffer(int width, int height) {
    size_t n = (size_t)width * height;
    std::vector<uint32_t> init(n * 2, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            init[(size_t)y * width + x] = seedCell(x, y); // half 0 = src
    GLuint ssbo = 0;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, init.size() * sizeof(uint32_t), init.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    return ssbo;
}

// One simulation step. `cur` (0/1) names the half holding current state; on
// return the new state lives in the other half (caller flips `cur`).
static void step(GLuint compute, GLuint ssbo, int width, int height, int cur) {
    int other = 1 - cur;
    size_t halfBytes = (size_t)width * height * sizeof(uint32_t);
    // copy cur -> other
    glBindBuffer(GL_COPY_READ_BUFFER, ssbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                        (GLintptr)(cur * halfBytes), (GLintptr)(other * halfBytes), (GLsizeiptr)halfBytes);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    glUseProgram(compute);
    glUniform1i(glGetUniformLocation(compute, "uWidth"), width);
    glUniform1i(glGetUniformLocation(compute, "uHeight"), height);
    glUniform1i(glGetUniformLocation(compute, "uSrc"), cur);
    glUniform1i(glGetUniformLocation(compute, "uDst"), other);
    glDispatchCompute((width + 15) / 16, (height + 15) / 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

// ---------------------------------------------------------------------------
// Headless benchmark.
// ---------------------------------------------------------------------------
static int runBench(int steps, int width, int height) {
    GLFWwindow* win = initGL(false, 64, 64, "bench");
    if (!win) return 1;

    GLuint compute = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});
    GLuint ssbo = makeGridBuffer(width, height);

    int cur = 0;
    glFinish();
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) { step(compute, ssbo, width, height, cur); cur = 1 - cur; }
    glFinish();
    auto end = std::chrono::steady_clock::now();

    std::vector<uint32_t> out((size_t)width * height);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)(cur * (size_t)width * height * sizeof(uint32_t)),
                       (GLsizeiptr)(out.size() * sizeof(uint32_t)), out.data());

    uint64_t sand = 0;
    for (uint32_t v : out) sand += (v & 1u);
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    double cells = (double)width * height * steps;
    double mcells = (elapsedMs > 0.0) ? cells / (elapsedMs / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=opengl rule=gpu width=%d height=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx sand=%llu\n",
           width, height, steps, elapsedMs, mcells,
           (unsigned long long)checksum(out), (unsigned long long)sand);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

// ---------------------------------------------------------------------------
// Interactive mode.
// ---------------------------------------------------------------------------
static void uploadHalf(GLuint ssbo, int width, int height, int half, const std::vector<uint32_t>& host) {
    size_t halfBytes = (size_t)width * height * sizeof(uint32_t);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)(half * halfBytes), (GLsizeiptr)halfBytes, host.data());
}
static void downloadHalf(GLuint ssbo, int width, int height, int half, std::vector<uint32_t>& host) {
    size_t halfBytes = (size_t)width * height * sizeof(uint32_t);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)(half * halfBytes), (GLsizeiptr)halfBytes, host.data());
}

static int runInteractive(int width, int height) {
    int renderW = width * PIXEL_SIZE, renderH = height * PIXEL_SIZE;
    GLFWwindow* win = initGL(true, renderW, renderH, "OpenGL Sand Simulation");
    if (!win) return 1;
    glfwSwapInterval(1);

    GLuint compute = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});
    GLuint present = linkProgram({compileShader(GL_VERTEX_SHADER, kPresentVert),
                                  compileShader(GL_FRAGMENT_SHADER, kPresentFrag)});
    GLuint ssbo = makeGridBuffer(width, height);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);

    std::vector<uint32_t> host((size_t)width * height, 0);
    downloadHalf(ssbo, width, height, 0, host); // start from the seeded half
    int cur = 0;
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    bool prevC = false, prevR = false;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        bool cDown = glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS;
        bool rDown = glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS;
        if (cDown && !prevC) std::fill(host.begin(), host.end(), 0u);
        if (rDown && !prevR)
            for (auto& v : host) { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                                   v = ((rng >> 40) % 100u < 30u) ? 1u : 0u; }
        prevC = cDown; prevR = rDown;

        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            double mx, my; glfwGetCursorPos(win, &mx, &my);
            int cx = (int)mx / PIXEL_SIZE, cy = (int)my / PIXEL_SIZE, r = 5;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && dx*dx + dy*dy <= r*r)
                        host[(size_t)ny * width + nx] = 1u;
                }
        }

        uploadHalf(ssbo, width, height, cur, host);
        step(compute, ssbo, width, height, cur);
        cur = 1 - cur;
        downloadHalf(ssbo, width, height, cur, host);

        glViewport(0, 0, renderW, renderH);
        glUseProgram(present);
        glUniform1i(glGetUniformLocation(present, "uWidth"), width);
        glUniform1i(glGetUniformLocation(present, "uHeight"), height);
        glUniform1i(glGetUniformLocation(present, "uCur"), cur);
        glUniform1i(glGetUniformLocation(present, "uPixel"), PIXEL_SIZE);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
    }
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps  = (argc > 2) ? atoi(argv[2]) : 1000;
        int width  = (argc > 3) ? atoi(argv[3]) : 400;
        int height = (argc > 4) ? atoi(argv[4]) : 300;
        return runBench(steps, width, height);
    }
    return runInteractive(400, 300);
}
