// ===================== main.c (instrumented for Nsight Systems) =====================
// Adds:
//  - OS thread names (Linux): capture / encrypt / wr_enc / wr_dec
//  - Optional NVTX ranges/marks (enable with -DUSE_NVTX and -lnvToolsExt)
//
// Build example:
//   g++ main.c ... -DUSE_NVTX -lnvToolsExt
//
// Profile example:
//   nsys profile -t cuda,nvtx,osrt --force-overwrite=true -o run ./your_app

#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <cuda_runtime.h>

#include "../include/encrypt_kernel.hpp"
#include "../include/prbg_main_plcm.hpp"
#include "../include/confusion_kernel.hpp"
#include "../include/diffusion_kernel.hpp"
#include "../include/utils.hpp"

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

// ===== Thread naming (Linux) + NVTX (Nsight Systems) =====
#include <pthread.h>
#include <sys/prctl.h>

#ifdef USE_NVTX
  #include <nvToolsExt.h>
#endif

static inline void set_os_thread_name(const char* name) {
    // Linux thread name limit is 16 bytes including '\0'
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", name);
    prctl(PR_SET_NAME, buf, 0, 0, 0);
    pthread_setname_np(pthread_self(), buf);
}

#ifdef USE_NVTX
static inline void nvtx_name_this_thread(const char* name) {
    // Nsight Systems will show this in the timeline
    nvtxNameOsThreadA(pthread_self(), name);
}

struct NvtxRange {
    NvtxRange(const char* msg) { nvtxRangePushA(msg); }
    ~NvtxRange() { nvtxRangePop(); }
};

#define NVTX_RANGE(name) NvtxRange _nvtx_range_##__LINE__(name)
#define NVTX_MARK(msg)   nvtxMarkA(msg)
#else
static inline void nvtx_name_this_thread(const char*) {}
#define NVTX_RANGE(name) do {} while(0)
#define NVTX_MARK(msg)   do {} while(0)
#endif

#define ROUNDS 5
#define NUM_CHANNELS 3

// ===================== Runtime-detected video parameters =====================
static int gW = 0;
static int gH = 0;
static int gFPS = 20;

// ===================== Subframe settings =====================
// Choose ONE pair:
static constexpr int SUB_W = 12;
static constexpr int SUB_H = 8;
// static constexpr int SUB_W = 32;
// static constexpr int SUB_H = 32;

// -------------------- Packet --------------------
struct FramePacket {
    cv::Mat frame;
    int id;
};

// -------------------- Queues --------------------
static std::queue<FramePacket> q_in;
static std::queue<FramePacket> q_out_enc;   // encrypted frames
static std::queue<FramePacket> q_out_dec;   // decrypted frames

// -------------------- Sync --------------------
static std::mutex m_in, m_enc, m_dec;
static std::condition_variable cv_in, cv_in_space;
static std::condition_variable cv_enc, cv_enc_space;
static std::condition_variable cv_dec, cv_dec_space;

static std::atomic<bool> capture_done(false);
static std::atomic<bool> encrypt_done(false);

static const int MAX_QUEUE_IN  = 4;
static const int MAX_QUEUE_OUT = 4;

// -------------------- CUDA Helpers --------------------
static inline void CHECK_CUDA(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "\n[CUDA ERROR] " << msg << " : " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// -------------------- Helpers Decl --------------------
void PRBGA_init(int numParameters,
                double ** d_keysAndControlPs,
                unsigned char ** d_values4ByteStream_1,
                unsigned char ** d_values4ByteStream_2,
                unsigned char ** d_byteStreamFinal,
                int PRBGAiterations);

void device_frame_allocation(int total_pixels, unsigned char ** d_sd_array, int numSubFrames);
void freeMemory(unsigned char *d_byteStreamFinal);
static int computeBitrate(int w, int h, int fps);

// -------------------- Threads --------------------
void capture_thread(cv::VideoCapture& cap);
void encrypt_thread();
void writer_enc_thread(cv::VideoWriter& writer);
void writer_dec_thread(cv::VideoWriter& writer);

// =====================================================
//                      MAIN
// =====================================================
int main(int argc, char** argv) {

    std::string inputPath     = "testVideos/960x960_20fps.mp4";
    //std::string inputPath     = "testVideos/768x768_20fps.mp4";
    //std::string inputPath     = "testVideos/672x672_20fps.mp4";
    //std::string inputPath     = "testVideos/576x576_20fps.mp4";
    //std::string inputPath     = "testVideos/480x480_20fps.mp4";
    //std::string inputPath     = "testVideos/384x384_20fps.mp4";
    //std::string inputPath     = "testVideos/288x288_20fps.mp4";
    //std::string inputPath     = "testVideos/192x192_20fps.mp4";
    //std::string inputPath       = "testVideos/96x96_20fps.mp4";

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open input video: " << inputPath << "\n";
        return -1;
    }

    // ---- detect input size + fps ----
    gW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    gH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::string outEncPath = "testResults/encrypted_output" + std::to_string(gW) + "x" + std::to_string(gH) + ".mp4";
    std::string outDecPath = "testResults/decrypted_output" + std::to_string(gW) + "x" + std::to_string(gH) + ".mp4";

    double fps_d = cap.get(cv::CAP_PROP_FPS);
    gFPS = (int)std::round(fps_d);
    printf("gfps: %d\n", gFPS);
    if (gFPS <= 0) gFPS = 20;  // fallback

    if (gW <= 0 || gH <= 0) {
        std::cerr << "ERROR: Invalid input dimensions\n";
        return -1;
    }

    // ---- guard: must be multiple of subframe size ----
    if ((gW % SUB_W) != 0 || (gH % SUB_H) != 0) {
        std::cerr << "ERROR: Input size not multiple of subframe size\n";
        std::cerr << "Input = " << gW << "x" << gH
                  << " | subframe = " << SUB_W << "x" << SUB_H << "\n";
        return -1;
    }

    std::cout << "[main] input: " << gW << "x" << gH << " @ " << gFPS
              << " FPS | subframe " << SUB_W << "x" << SUB_H << "\n";

    const int width  = gW;
    const int height = gH;
    const int fps    = gFPS;

    int bitrate = computeBitrate(width, height, fps);
    std::cout << "[main] bitrate = " << bitrate << " bps\n";

    // ---- ENCRYPTED OUTPUT (NVENC) ----
    std::string pipeline_enc =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "nvv4l2h264enc bitrate=" + std::to_string(bitrate) +
        " control-rate=1 insert-sps-pps=true iframeinterval=30 maxperf-enable=1 ! "
        "h264parse ! qtmux ! filesink location=" + outEncPath + " sync=false";

    cv::VideoWriter writer_enc(pipeline_enc, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);
    if (!writer_enc.isOpened()) {
        std::cerr << "ERROR: Could not open GStreamer VideoWriter (encrypted)\n";
        return -1;
    }

    // ---- DECRYPTED OUTPUT (NVENC) ----
    std::string pipeline_dec =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "nvv4l2h264enc bitrate=" + std::to_string(bitrate) +
        " control-rate=1 insert-sps-pps=true iframeinterval=30 maxperf-enable=1 ! "
        "h264parse ! qtmux ! filesink location=" + outDecPath + " sync=false";

    cv::VideoWriter writer_dec(pipeline_dec, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);
    if (!writer_dec.isOpened()) {
        std::cerr << "ERROR: Could not open GStreamer VideoWriter (decrypted)\n";
        return -1;
    }

    // ---- Launch threads ----
    std::thread t1(capture_thread, std::ref(cap));
    std::thread t2(encrypt_thread);
    std::thread t3(writer_enc_thread, std::ref(writer_enc));
    std::thread t4(writer_dec_thread, std::ref(writer_dec));

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    std::cout << "[main] finished cleanly \n";
    return 0;
}

// =====================================================
//                  CAPTURE thread
// =====================================================
void capture_thread(cv::VideoCapture& cap) {
    set_os_thread_name("capture");
    nvtx_name_this_thread("capture");
    NVTX_MARK("capture: start");

    std::cout << "[capture] started\n";

    int id = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        // enforce BGR 8UC3
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        if (frame.depth() != CV_8U) {
            frame.convertTo(frame, CV_8U);
        }

        if (frame.type() != CV_8UC3) {
            frame = frame.clone(); // safest layout
        }

        if (!frame.isContinuous()) {
            frame = frame.clone();
        }

        {
            NVTX_RANGE("capture: wait space + push q_in");
            std::unique_lock<std::mutex> lk(m_in);
            cv_in_space.wait(lk, [] { return (int)q_in.size() < MAX_QUEUE_IN; });
            q_in.push({ frame.clone(), id++ });
        }
        cv_in.notify_one();

        if ((id % 60) == 0) NVTX_MARK("capture: milestone");
    }

    capture_done = true;
    cv_in.notify_all();
    NVTX_MARK("capture: done");
    std::cout << "[capture] done\n";
}

// =====================================================
//                   ENCRYPT + DECRYPT
// =====================================================
void encrypt_thread() {
    set_os_thread_name("encrypt");
    nvtx_name_this_thread("encrypt");
    NVTX_MARK("encrypt: start");

    std::cout << "[encrypt] started\n";

    const int width  = gW;
    const int height = gH;

    const int subframeWidth  = SUB_W;
    const int subframeHeight = SUB_H;

    const int PRBGAsAmount = (width * height) / 6;

    // PRBGAiterations must not be 0
    int PRBGAiterations =
        (int)std::ceil(((float)(subframeHeight * subframeWidth * NUM_CHANNELS) / 6.0f)) / 16;
    PRBGAiterations = std::max(1, PRBGAiterations);

    const int numSubFrames = (width * height) / (subframeHeight * subframeWidth);

    const double globalKey = 0.223456;
    const double p = 0.3;

    const size_t frameBytes = (size_t)width * height * NUM_CHANNELS * sizeof(unsigned char);

    // PRBGA / bytestream
    std::vector<unsigned char> byteStreamFinal;
    unsigned char *d_values4ByteStream_1 = nullptr;
    unsigned char *d_values4ByteStream_2 = nullptr;
    unsigned char *d_byteStreamFinal     = nullptr;
    double *d_keysAndControlPs           = nullptr;

    // diffusion support
    unsigned char *d_sd_array = nullptr;

    // ping-pong GPU buffers
    unsigned char *d_input[2]  = {nullptr, nullptr};
    unsigned char *d_output[2] = {nullptr, nullptr};

    // pinned host buffers (encrypted + decrypted)
    unsigned char *h_enc_pinned[2] = {nullptr, nullptr};
    unsigned char *h_dec_pinned[2] = {nullptr, nullptr};

    double *keysAndControlPsPinned = nullptr;
    int buffer_id[2] = {-1, -1};

    // CUDA stream + events (only for encrypted completion)
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1), "cudaStreamCreate");

    cudaEvent_t encStart[2], encStop[2];

    CHECK_CUDA(cudaEventCreate(&encStart[0]), "cudaEventCreate encStart0");
    CHECK_CUDA(cudaEventCreate(&encStart[1]), "cudaEventCreate encStart1");
    CHECK_CUDA(cudaEventCreate(&encStop[0]),  "cudaEventCreate encStop0");
    CHECK_CUDA(cudaEventCreate(&encStop[1]),  "cudaEventCreate encStop1");

    // allocate buffers
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMallocHost(&h_enc_pinned[i], frameBytes), "cudaMallocHost h_enc_pinned");
        CHECK_CUDA(cudaMallocHost(&h_dec_pinned[i], frameBytes), "cudaMallocHost h_dec_pinned");

        CHECK_CUDA(cudaMalloc(&d_input[i],  frameBytes), "cudaMalloc d_input");
        CHECK_CUDA(cudaMalloc(&d_output[i], frameBytes), "cudaMalloc d_output");
    }

    CHECK_CUDA(cudaMallocHost(&keysAndControlPsPinned, (2 * PRBGAsAmount + 1) * sizeof(double)),
               "cudaMallocHost keysAndControlPsPinned");

    {
        NVTX_RANGE("encrypt: PRBGA_init + frame alloc");
        PRBGA_init(2 * PRBGAsAmount,
                   &d_keysAndControlPs,
                   &d_values4ByteStream_1,
                   &d_values4ByteStream_2,
                   &d_byteStreamFinal,
                   PRBGAiterations);

        device_frame_allocation(width * height, &d_sd_array, numSubFrames);
    }

    int frameCount = 0;
    uint64_t sc = 0;

    double enc_sum_ms = 0.0;
    int enc_measured_frames = 0;

    while (true) {
        FramePacket pkt;

        // ---- pop input ----
        {
            NVTX_RANGE("encrypt: wait+pop q_in");
            std::unique_lock<std::mutex> lk(m_in);
            cv_in.wait(lk, [] { return !q_in.empty() || capture_done.load(); });

            if (q_in.empty() && capture_done.load())
                break;

            pkt = q_in.front();
            q_in.pop();
        }
        cv_in_space.notify_one();

        const int cur  = frameCount % 2;
        const int prev = (frameCount + 1) % 2;

        buffer_id[cur] = pkt.id;

        // ---- PRBG main keys (CPU) ----
        {
            NVTX_RANGE("encrypt: PRBG CPU");
            generatePRBGMainKeysv3Pinned(globalKey, p, 2 * PRBGAsAmount + 1, &sc, keysAndControlPsPinned);
        }

        // ---- copy input frame -> pinned buffer ----
        {
            NVTX_RANGE("encrypt: memcpy host->pinned");
            std::memcpy(h_enc_pinned[cur], pkt.frame.data, frameBytes);
        }

        CHECK_CUDA(cudaEventRecord(encStart[cur], stream1), "cudaEventRecord(encStart)");

        // ---- PRBGA + bytestream generation ----
        {
            NVTX_RANGE("encrypt: PRBGA+ByteStream (GPU)");
            PRBGAandByteStreamGenWrapper(
                d_keysAndControlPs,
                keysAndControlPsPinned,
                (2 * PRBGAsAmount),
                d_values4ByteStream_1,
                d_values4ByteStream_2,
                d_byteStreamFinal,
                byteStreamFinal,
                PRBGAiterations,
                subframeHeight,
                subframeWidth,
                width,
                height,
                stream1
            );
        }

        // ---- H2D ----
        {
            NVTX_RANGE("encrypt: H2D");
            CHECK_CUDA(cudaMemcpyAsync(d_input[cur], h_enc_pinned[cur], frameBytes,
                                       cudaMemcpyHostToDevice, stream1),
                       "cudaMemcpyAsync H2D");
        }

        // ===================== FORWARD ENCRYPT =====================
        int invConf = 0;
        int invDiff = 0;

        {
            NVTX_RANGE("encrypt: ENCRYPT rounds");
            for (int r = 0; r < ROUNDS; r++) {
                confusionOpWrapper(pkt.frame.data, d_input[cur], pkt.frame.data, d_output[cur],
                                   width, height, subframeHeight, sc, invConf, stream1);
                std::swap(d_input[cur], d_output[cur]);

                diffusionOpWrapper(d_byteStreamFinal, d_input[cur], pkt.frame.data, d_output[cur],
                                   pkt.frame.data, width, height, subframeHeight, subframeWidth,
                                   invDiff, d_sd_array, stream1);
                std::swap(d_input[cur], d_output[cur]);
            }
        }

        if (frameCount % 30 == 0) {
            std::cout << "." << std::flush;
            NVTX_MARK("encrypt: milestone");
        }

        // ---- D2H encrypted ----
        {
            NVTX_RANGE("encrypt: D2H(enc)");
            CHECK_CUDA(cudaMemcpyAsync(h_enc_pinned[cur], d_input[cur], frameBytes,
                                       cudaMemcpyDeviceToHost, stream1),
                       "cudaMemcpyAsync D2H(enc)");
        }

        CHECK_CUDA(cudaEventRecord(encStop[cur], stream1), "cudaEventRecord(encStop)");

        // ---- push PREVIOUS encrypted frame ----
        if (frameCount >= 1) {
            {
                NVTX_RANGE("encrypt: sync prev encStop");
                CHECK_CUDA(cudaEventSynchronize(encStop[prev]), "cudaEventSync(prev encStop)");
            }

            float enc_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&enc_ms, encStart[prev], encStop[prev]),
                       "cudaEventElapsedTime(prev enc)");

            enc_sum_ms += (double)enc_ms;
            enc_measured_frames++;

            cv::Mat out_enc(height, width, CV_8UC3);
            std::memcpy(out_enc.data, h_enc_pinned[prev], frameBytes);

            {
                NVTX_RANGE("encrypt: wait space + push q_out_enc");
                std::unique_lock<std::mutex> lk(m_enc);
                cv_enc_space.wait(lk, [] { return (int)q_out_enc.size() < MAX_QUEUE_OUT; });
                q_out_enc.push({ out_enc, buffer_id[prev] });
            }
            cv_enc.notify_one();
        }

        // ===================== INVERSE DECRYPT CHECK =====================
        invConf = 1;
        invDiff = 1;

        {
            NVTX_RANGE("encrypt: DECRYPT rounds");
            for (int r = ROUNDS - 1; r >= 0; r--) {
                diffusionOpWrapper(d_byteStreamFinal, d_input[cur], pkt.frame.data, d_output[cur],
                                   pkt.frame.data, width, height, subframeHeight, subframeWidth,
                                   invDiff, d_sd_array, stream1);
                std::swap(d_input[cur], d_output[cur]);

                confusionOpWrapper(pkt.frame.data, d_input[cur], pkt.frame.data, d_output[cur],
                                   width, height, subframeHeight, sc, invConf, stream1);
                std::swap(d_input[cur], d_output[cur]);
            }
        }

        // ---- D2H decrypted ----
        {
            NVTX_RANGE("encrypt: D2H(dec)");
            CHECK_CUDA(cudaMemcpyAsync(h_dec_pinned[cur], d_input[cur], frameBytes,
                                       cudaMemcpyDeviceToHost, stream1),
                       "cudaMemcpyAsync D2H(dec)");
        }

        // ---- push decrypted frame ----
        {
            cv::Mat out_dec(height, width, CV_8UC3);
            std::memcpy(out_dec.data, h_dec_pinned[cur], frameBytes);

            NVTX_RANGE("encrypt: wait space + push q_out_dec");
            std::unique_lock<std::mutex> lk(m_dec);
            cv_dec_space.wait(lk, [] { return (int)q_out_dec.size() < MAX_QUEUE_OUT; });
            q_out_dec.push({ out_dec, pkt.id });
        }
        cv_dec.notify_one();

        frameCount++;
    }

    if (enc_measured_frames > 0) {
        double enc_avg_ms = enc_sum_ms / enc_measured_frames;
        std::cout << "\n =============FINAL GPU ENCRYPT REPORT ============\n";
        std::cout << "Measured frames      : " << enc_measured_frames << "\n";
        std::cout << "Avg enc time/frame   : " << enc_avg_ms << " ms\n";
        std::cout << "Avg theoretical FPS  : " << (1000.0 / enc_avg_ms) << "\n\n";
    }

    // ---- flush LAST encrypted frame ----
    if (frameCount > 0) {
        int last = (frameCount - 1) % 2;

        {
            NVTX_RANGE("encrypt: sync last encStop");
            CHECK_CUDA(cudaEventSynchronize(encStop[last]), "cudaEventSync(last encStop)");
        }

        float enc_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&enc_ms, encStart[last], encStop[last]),
                   "cudaEventElapsedTime(last enc)");
        std::cout << "[GPU] last enc time = " << enc_ms << " ms\n";

        cv::Mat out_enc(height, width, CV_8UC3);
        std::memcpy(out_enc.data, h_enc_pinned[last], frameBytes);

        {
            NVTX_RANGE("encrypt: wait space + push q_out_enc (last)");
            std::unique_lock<std::mutex> lk(m_enc);
            cv_enc_space.wait(lk, [] { return (int)q_out_enc.size() < MAX_QUEUE_OUT; });
            q_out_enc.push({ out_enc, buffer_id[last] });
        }
        cv_enc.notify_one();
    }

    encrypt_done = true;
    cv_enc.notify_all();
    cv_dec.notify_all();

    // ---- cleanup ----
    {
        NVTX_RANGE("encrypt: cleanup");
        cudaStreamDestroy(stream1);
        cudaEventDestroy(encStart[0]);
        cudaEventDestroy(encStart[1]);
        cudaEventDestroy(encStop[0]);
        cudaEventDestroy(encStop[1]);

        freeMemory(d_byteStreamFinal);

        for (int i = 0; i < 2; i++) {
            cudaFree(d_input[i]);
            cudaFree(d_output[i]);
            cudaFreeHost(h_enc_pinned[i]);
            cudaFreeHost(h_dec_pinned[i]);
        }

        cudaFreeHost(keysAndControlPsPinned);
        cudaFree(d_keysAndControlPs);
        cudaFree(d_values4ByteStream_1);
        cudaFree(d_values4ByteStream_2);
        cudaFree(d_sd_array);
    }

    NVTX_MARK("encrypt: done");
    std::cout << "[encrypt] done\n";
}

// =====================================================
//                WRITER (ENCRYPTED)
// =====================================================
void writer_enc_thread(cv::VideoWriter& writer) {
    set_os_thread_name("wr_enc");
    nvtx_name_this_thread("wr_enc");
    NVTX_MARK("wr_enc: start");

    std::cout << "[writer_enc] started\n";

    while (true) {
        FramePacket pkt;

        {
            NVTX_RANGE("wr_enc: wait+pop q_out_enc");
            std::unique_lock<std::mutex> lk(m_enc);
            cv_enc.wait(lk, [] { return !q_out_enc.empty() || encrypt_done.load(); });

            if (q_out_enc.empty() && encrypt_done.load())
                break;

            pkt = q_out_enc.front();
            q_out_enc.pop();
        }
        cv_enc_space.notify_one();

        {
            NVTX_RANGE("wr_enc: writer.write");
            writer.write(pkt.frame);
        }
    }

    NVTX_MARK("wr_enc: done");
    std::cout << "[writer_enc] done\n";
}

// =====================================================
//                WRITER (DECRYPTED)
// =====================================================
void writer_dec_thread(cv::VideoWriter& writer) {
    set_os_thread_name("wr_dec");
    nvtx_name_this_thread("wr_dec");
    NVTX_MARK("wr_dec: start");

    std::cout << "[writer_dec] started\n";

    while (true) {
        FramePacket pkt;

        {
            NVTX_RANGE("wr_dec: wait+pop q_out_dec");
            std::unique_lock<std::mutex> lk(m_dec);
            cv_dec.wait(lk, [] { return !q_out_dec.empty() || encrypt_done.load(); });

            if (q_out_dec.empty() && encrypt_done.load())
                break;

            pkt = q_out_dec.front();
            q_out_dec.pop();
        }
        cv_dec_space.notify_one();

        {
            NVTX_RANGE("wr_dec: writer.write");
            writer.write(pkt.frame);
        }
    }

    NVTX_MARK("wr_dec: done");
    std::cout << "[writer_dec] done\n";
}

// =====================================================
//                   ALLOC HELPERS
// =====================================================
void PRBGA_init(int numParameters,
                double ** d_keysAndControlPs,
                unsigned char ** d_values4ByteStream_1,
                unsigned char ** d_values4ByteStream_2,
                unsigned char ** d_byteStreamFinal,
                int PRBGAiterations) {

    int numKeys = numParameters / 2;
    int totalSize = numKeys * PRBGAiterations * 6;

    cudaMalloc((void**)d_keysAndControlPs,     numParameters * sizeof(double));
    cudaMalloc((void**)d_values4ByteStream_1,  totalSize * sizeof(unsigned char));
    cudaMalloc((void**)d_values4ByteStream_2,  totalSize * sizeof(unsigned char));
    cudaMalloc((void**)d_byteStreamFinal,      totalSize * sizeof(unsigned char));
}

void device_frame_allocation(int total_pixels, unsigned char ** d_sd_array, int numSubFrames) {
    cudaMalloc((void**)d_sd_array, numSubFrames * sizeof(unsigned char));
}

void freeMemory(unsigned char *d_byteStreamFinal) {
    cudaFree(d_byteStreamFinal);
}

static int computeBitrate(int w, int h, int fps) {
    const double bpp = 0.15;
    double bitrate = (double)w * (double)h * (double)fps * bpp;

    // Safe clamp for Jetson NVENC stability
    const double minBps = 300000;     // 300 kbps
    const double maxBps = 20000000;   // 20 Mbps

    if (bitrate < minBps) bitrate = minBps;
    if (bitrate > maxBps) bitrate = maxBps;

    return (int)bitrate;
}
