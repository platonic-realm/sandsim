/*
 * sandsim - multi-material streaming world, OpenGL compute (bit-identical to CPU)
 *
 * The same order-independent rule as cpp/simd_core.h, on the GPU. Each frame is
 * 16 compute dispatches (one per disjoint sub-pass) with a memory barrier
 * between. A thread decides whether its cell is the SOURCE of a swap purely from
 * its (x,y) coordinates (parity / boundary pattern) -- never from cell content --
 * so non-source threads touch nothing and the in-place update is race-free and
 * reproduces the CPU result exactly.
 *
 * Huge worlds stream like the CPU build: the live GW x GH window lives in a GPU
 * buffer; a CPU shadow drives the identical chunk<->disk logic. Buffer transfers
 * happen only when the camera moves (and at summary) -- the per-frame simulation
 * stays entirely on the GPU. Same seed + same camera path => same world, so the
 * whole-world checksum matches the C++ and Vulkan builds bit-for-bit.
 *
 * Modes:
 *   --bench [steps] [wch] [hch]   headless streaming benchmark (one RESULT line)
 *   (default)                     interactive viewer (arrows pan, number keys paint)
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

enum Material : uint8_t { EMPTY = 0, WALL = 1, SAND = 2, WATER = 3, GAS = 4, MATERIAL_COUNT = 5 };
enum { SG_DOWN, SG_GAS, SG_HORIZ };

static constexpr int CHUNK = 64;
static constexpr int GW = 4, GH = 4;
static constexpr int LW = GW * CHUNK, LH = GH * CHUNK;     // 256 x 256 live cells
static constexpr int PAD = 16;
static constexpr int SW = LW + 2 * PAD, SH = LH + 2 * PAD; // padded stride / height
static constexpr int X0 = PAD, X1 = PAD + LW, Y0 = PAD, Y1 = PAD + LH;

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

// The 16 sub-passes, in the exact order of cpp/simd_core.h.
struct Pass { int type, dx, dy, parity, grp; };   // type: 0 vert, 1 diag, 2 horiz
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

// --------------------------------------------------------------------------- shaders
static const char* kComputeSrc = R"GLSL(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Cells { uint cells[]; };
layout(std430, binding = 1) buffer Moved { uint moved[]; };
uniform int uSW, uX0, uX1, uY0, uY1;
uniform int uType, uDx, uDy, uParity, uGrp;
bool canEnter(uint s, uint t) {
    if (t == 1u) return false;                 // WALL
    if (s == 2u) return t==0u||t==3u||t==4u;   // SAND  -> E,W,G
    if (s == 3u) return t==0u||t==4u;          // WATER -> E,G
    if (s == 4u) return t==0u;                 // GAS   -> E
    return false;
}
bool eligible(uint s) {
    if (uGrp == 0) return s==2u||s==3u;        // DOWN: sand,water
    if (uGrp == 1) return s==4u;               // GAS
    return s==3u||s==4u;                        // HORIZ: water,gas
}
void main() {
    int x = uX0 + int(gl_GlobalInvocationID.x);
    int y = uY0 + int(gl_GlobalInvocationID.y);
    if (x >= uX1 || y >= uY1) return;
    int cx = x - uX0;
    bool src = (uType == 0) ? (((y - uY0) & 1) == uParity)   // vertical: row parity
                            : ((cx & 1) == uParity);          // diag/horiz: column parity
    if (uType == 2) {                                         // horizontal boundary skip
        if (uDx < 0 && uParity == 0 && (cx % 16) == 0)  src = false;
        if (uDx > 0 && uParity == 1 && (cx % 16) == 15) src = false;
    }
    if (!src) return;
    int si = y * uSW + x;
    int ti = (y + uDy) * uSW + (x + uDx);
    uint s = cells[si], t = cells[ti];
    if (eligible(s) && canEnter(s, t) && moved[si] == 0u && moved[ti] == 0u) {
        cells[ti] = s; cells[si] = t;
        moved[si] = 1u; moved[ti] = 1u;
    }
}
)GLSL";

static const char* kPresentVert = R"GLSL(
#version 430
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
)GLSL";

static const char* kPresentFrag = R"GLSL(
#version 430
layout(std430, binding = 0) buffer Cells { uint cells[]; };
uniform int uSW, uX0, uY0, uLW, uLH, uRW, uRH;
out vec4 frag;
void main() {
    int px = int(gl_FragCoord.x);
    int py = uRH - 1 - int(gl_FragCoord.y);
    int lx = px * uLW / uRW;
    int ly = py * uLH / uRH;
    uint m = cells[(uY0 + ly) * uSW + (uX0 + lx)] & 0xFFu;
    vec3 c = vec3(0.0);
    if      (m == 1u) c = vec3(0.502, 0.502, 0.502);
    else if (m == 2u) c = vec3(0.886, 0.784, 0.471);
    else if (m == 3u) c = vec3(0.267, 0.533, 1.000);
    else if (m == 4u) c = vec3(0.690, 0.769, 0.871);
    frag = vec4(c, 1.0);
}
)GLSL";

static GLuint compileShader(GLenum kind, const char* src) {
    GLuint sh = glCreateShader(kind);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof(log), nullptr, log); fprintf(stderr, "shader: %s\n", log); exit(1); }
    return sh;
}
static GLuint linkProgram(std::vector<GLuint> shaders) {
    GLuint p = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    for (GLuint s : shaders) glDeleteShader(s);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof(log), nullptr, log); fprintf(stderr, "link: %s\n", log); exit(1); }
    return p;
}

static GLFWwindow* initGL(bool visible, int w, int h) {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    if (!glfwInit()) { glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM); if (!glfwInit()) return nullptr; }
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(w, h, "sandsim (OpenGL)", nullptr, nullptr);
    if (!win) { fprintf(stderr, "no OpenGL 4.3 context\n"); glfwTerminate(); return nullptr; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    GLenum ge = glewInit();
    if (ge != GLEW_OK && ge != GLEW_ERROR_NO_GLX_DISPLAY) { fprintf(stderr, "glewInit: %s\n", glewGetErrorString(ge)); glfwTerminate(); return nullptr; }
    glGetError();
    return win;
}

// --------------------------------------------------------------------------- world
// Mirrors cpp/sandsim_world.cpp's SimdWorld: a CPU shadow drives identical
// chunk<->disk streaming; the live window also lives in a GPU buffer, where the
// 16-pass step runs. Shadow and GPU buffer are synced only at window changes
// (and summary) -- not per frame -- so the heavy loop stays on the GPU.
class GpuWorld {
public:
    GpuWorld(int wbox, int hbox, std::string dir, GLuint computeProg)
        : wbox(wbox), hbox(hbox), dir(std::move(dir)), prog(computeProg) {
        std::filesystem::create_directories(this->dir);
        shadow.assign((size_t)SW * SH, WALL);

        glGenBuffers(1, &cellsBuf); glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellsBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (size_t)SW * SH * 4, shadow.data(), GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cellsBuf);
        glGenBuffers(1, &movedBuf); glBindBuffer(GL_SHADER_STORAGE_BUFFER, movedBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (size_t)SW * SH * 4, nullptr, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, movedBuf);

        glUseProgram(prog);
        lSW = glGetUniformLocation(prog, "uSW"); lX0 = glGetUniformLocation(prog, "uX0");
        lX1 = glGetUniformLocation(prog, "uX1"); lY0 = glGetUniformLocation(prog, "uY0");
        lY1 = glGetUniformLocation(prog, "uY1"); lType = glGetUniformLocation(prog, "uType");
        lDx = glGetUniformLocation(prog, "uDx"); lDy = glGetUniformLocation(prog, "uDy");
        lPar = glGetUniformLocation(prog, "uParity"); lGrp = glGetUniformLocation(prog, "uGrp");
        glUniform1i(lSW, SW); glUniform1i(lX0, X0); glUniform1i(lX1, X1);
        glUniform1i(lY0, Y0); glUniform1i(lY1, Y1);
    }
    ~GpuWorld() { glDeleteBuffers(1, &cellsBuf); glDeleteBuffers(1, &movedBuf); }

    GLuint buffer() const { return cellsBuf; }

    void generateAllToDisk() {
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) { genBox(cx, cy, buf); writeBox(cx, cy, buf); }
    }

    void setWindow(int camCx, int camCy) {
        if (windowValid && camCx == winCx && camCy == winCy) return;
        syncDown();                                // bring GPU state into the shadow
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        if (windowValid)
            for (int gy = 0; gy < GH; ++gy)
                for (int gx = 0; gx < GW; ++gx) { extractChunk(gx, gy, buf); writeBox(winCx + gx, winCy + gy, buf); }
        for (int gy = 0; gy < GH; ++gy)
            for (int gx = 0; gx < GW; ++gx) {
                if (!readBox(camCx + gx, camCy + gy, buf)) genBox(camCx + gx, camCy + gy, buf);
                injectChunk(gx, gy, buf);
            }
        winCx = camCx; winCy = camCy; windowValid = true;
        residentMax = GW * GH;
        syncUp();                                  // push the new window to the GPU
    }

    void step() {
        glUseProgram(prog);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, movedBuf);
        const uint32_t zero = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        for (const Pass& p : kPasses) {
            glUniform1i(lType, p.type); glUniform1i(lDx, p.dx); glUniform1i(lDy, p.dy);
            glUniform1i(lPar, p.parity); glUniform1i(lGrp, p.grp);
            glDispatchCompute(LW / 16, LH / 16, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        gpuAhead = true;
    }

    void paint(int lx, int ly, uint8_t material, int radius) {
        syncDown();
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = lx + dx, ny = ly + dy;
                if (nx >= 0 && nx < LW && ny >= 0 && ny < LH && dx * dx + dy * dy <= radius * radius)
                    shadow[(size_t)(ny + Y0) * SW + (nx + X0)] = material;
            }
        syncUp();
    }

    void summary(uint64_t& checksum, uint64_t counts[MATERIAL_COUNT]) {
        syncDown();
        for (int i = 0; i < MATERIAL_COUNT; ++i) counts[i] = 0;
        std::vector<uint8_t> buf((size_t)CHUNK * CHUNK);
        uint64_t c = 14695981039346656037ull;
        for (int cy = 0; cy < hbox; ++cy)
            for (int cx = 0; cx < wbox; ++cx) {
                bool inWin = windowValid && cx >= winCx && cx < winCx + GW && cy >= winCy && cy < winCy + GH;
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

private:
    int wbox, hbox;
    std::string dir;
    GLuint prog, cellsBuf = 0, movedBuf = 0;
    std::vector<uint32_t> shadow;        // CPU mirror of the live padded window
    bool gpuAhead = false;               // GPU buffer holds newer data than the shadow
    int winCx = 0, winCy = 0;
    bool windowValid = false;
    int residentMax = 0;
    long long nWrites = 0, nReads = 0;
    GLint lSW, lX0, lX1, lY0, lY1, lType, lDx, lDy, lPar, lGrp;

    void syncDown() {
        if (!gpuAhead) return;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellsBuf);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (size_t)SW * SH * 4, shadow.data());
        gpuAhead = false;
    }
    void syncUp() {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellsBuf);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (size_t)SW * SH * 4, shadow.data());
        gpuAhead = false;
    }

    void extractChunk(int gx, int gy, std::vector<uint8_t>& out) const {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                out[ly * CHUNK + lx] = (uint8_t)(shadow[(size_t)(Y0 + gy * CHUNK + ly) * SW + (X0 + gx * CHUNK + lx)] & 0xFFu);
    }
    void injectChunk(int gx, int gy, const std::vector<uint8_t>& in) {
        for (int ly = 0; ly < CHUNK; ++ly)
            for (int lx = 0; lx < CHUNK; ++lx)
                shadow[(size_t)(Y0 + gy * CHUNK + ly) * SW + (X0 + gx * CHUNK + lx)] = in[ly * CHUNK + lx];
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
    if (wbox < GW) wbox = GW;
    if (hbox < GH) hbox = GH;
    GLFWwindow* win = initGL(false, 64, 64);
    if (!win) return 1;
    GLuint prog = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});

    std::string dir = "/tmp/sandsim_world_gl_" + std::to_string(steps) + "_" +
                      std::to_string(wbox) + "x" + std::to_string(hbox);
    std::filesystem::remove_all(dir);
    GpuWorld world(wbox, hbox, dir, prog);
    world.generateAllToDisk();

    uint64_t startCk, startCnt[MATERIAL_COUNT];
    world.summary(startCk, startCnt);

    int nposX = wbox - GW + 1, nposY = hbox - GH + 1, nWin = nposX * nposY;
    (void)nposY;
    glFinish();
    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        int visit = (int)((long long)s * nWin / steps);
        if (visit >= nWin) visit = nWin - 1;
        int row = visit / nposX, col = visit % nposX;
        world.setWindow((row % 2 == 0) ? col : (nposX - 1 - col), row);
        world.step();
    }
    glFinish();
    auto end = std::chrono::steady_clock::now();

    uint64_t ck, cnt[MATERIAL_COUNT];
    world.summary(ck, cnt);
    bool conserved = true;
    for (int i = WALL; i <= GAS; ++i) if (cnt[i] != startCnt[i]) conserved = false;
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double mc = (ms > 0.0) ? (double)LW * LH * steps / (ms / 1000.0) / 1e6 : 0.0;
    printf("RESULT impl=opengl rule=world window=%dx%d wbox=%d hbox=%d steps=%d "
           "elapsed_ms=%.3f mcells_per_s=%.2f checksum=%016llx "
           "empty=%llu wall=%llu sand=%llu water=%llu gas=%llu "
           "resident_max=%d disk_writes=%lld disk_reads=%lld conserved=%s\n",
           GW, GH, wbox, hbox, steps, ms, mc, (unsigned long long)ck,
           (unsigned long long)cnt[EMPTY], (unsigned long long)cnt[WALL],
           (unsigned long long)cnt[SAND], (unsigned long long)cnt[WATER], (unsigned long long)cnt[GAS],
           world.residentMaxCount(), world.diskWrites(), world.diskReads(), conserved ? "yes" : "no");
    std::filesystem::remove_all(dir);
    glfwDestroyWindow(win); glfwTerminate();
    return conserved ? 0 : 2;
}

static int runInteractive() {
    static const int PIXEL = 2, WBOX = 16, HBOX = 16;
    int renderW = LW * PIXEL, renderH = LH * PIXEL;
    GLFWwindow* win = initGL(true, renderW, renderH);
    if (!win) return 1;
    GLuint compute = linkProgram({compileShader(GL_COMPUTE_SHADER, kComputeSrc)});
    GLuint present = linkProgram({compileShader(GL_VERTEX_SHADER, kPresentVert),
                                  compileShader(GL_FRAGMENT_SHADER, kPresentFrag)});
    GLuint vao; glGenVertexArrays(1, &vao);

    std::string dir = "/tmp/sandsim_world_gl_interactive";
    std::filesystem::remove_all(dir);
    GpuWorld world(WBOX, HBOX, dir, compute);
    world.generateAllToDisk();
    int camCx = 0, camCy = 0;
    world.setWindow(camCx, camCy);
    uint8_t current = SAND;

    glUseProgram(present);
    glUniform1i(glGetUniformLocation(present, "uSW"), SW);
    glUniform1i(glGetUniformLocation(present, "uX0"), X0);
    glUniform1i(glGetUniformLocation(present, "uY0"), Y0);
    glUniform1i(glGetUniformLocation(present, "uLW"), LW);
    glUniform1i(glGetUniformLocation(present, "uLH"), LH);
    glUniform1i(glGetUniformLocation(present, "uRW"), renderW);
    glUniform1i(glGetUniformLocation(present, "uRH"), renderH);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        if (glfwGetKey(win, GLFW_KEY_0) == GLFW_PRESS) current = EMPTY;
        if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) current = WALL;
        if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) current = SAND;
        if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) current = WATER;
        if (glfwGetKey(win, GLFW_KEY_4) == GLFW_PRESS) current = GAS;
        static bool pL = false, pR = false, pU = false, pD = false;
        bool l = glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS;
        bool r = glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS;
        bool u = glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS;
        bool d = glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS;
        if (l && !pL && camCx > 0) camCx--;
        if (r && !pR && camCx < WBOX - GW) camCx++;
        if (u && !pU && camCy > 0) camCy--;
        if (d && !pD && camCy < HBOX - GH) camCy++;
        pL = l; pR = r; pU = u; pD = d;

        world.setWindow(camCx, camCy);
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            double mx, my; glfwGetCursorPos(win, &mx, &my);
            world.paint((int)mx / PIXEL, (int)my / PIXEL, current, 4);
        }
        world.step();

        glUseProgram(present);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, world.buffer());
        glViewport(0, 0, renderW, renderH);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
    }
    std::filesystem::remove_all(dir);
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 600;
        int wbox  = (argc > 3) ? std::atoi(argv[3]) : 6;
        int hbox  = (argc > 4) ? std::atoi(argv[4]) : 6;
        return runBench(steps, wbox, hbox);
    }
    return runInteractive();
}
