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
#include <cstring>   // memcpy
#include <algorithm> // std::max

#define ROUNDS 5
#define NUM_CHANNELS 3

// ===================== Dynamic runtime video params =====================
static int gW = 0;
static int gH = 0;
static int gFPS = 20;

// ===================== Subframe size (you can switch these) ==============
// Option A:
// static constexpr int SUB_W = 12;
// static constexpr int SUB_H = 8;

// Option B:
static constexpr int SUB_W = 32;
static constexpr int SUB_H = 32;

// -------------------- Packet --------------------
struct FramePacket {
    cv::Mat frame;
    int id;
};

// -------------------- Queues + Sync --------------------
static std::queue<FramePacket> q_in;
static std::queue<FramePacket> q_out;

static std::mutex m_in, m_out;
static std::condition_variable cv_in, cv_in_space;
static std::condition_variable cv_out, cv_out_space;

static std::atomic<bool> capture_done(false);
static std::atomic<bool> encrypt_done(false);

static const int MAX_QUEUE_IN  = 4;
static const int MAX_QUEUE_OUT = 4;

// -------------------- Threads --------------------
void capture_thread(cv::VideoCapture& cap);
void encrypt_thread();
void writer_thread(cv::VideoWriter& writer);

// -------------------- CUDA Helpers --------------------
static inline void CHECK_CUDA(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << " : " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// ---------- Local helpers (static so no link errors) ----------
static void PRBGA_init_local(int numParameters,
                             double ** d_keysAndControlPs,
                             unsigned char ** d_values4ByteStream_1,
                             unsigned char ** d_values4ByteStream_2,
                             unsigned char ** d_byteStreamFinal,
                             int PRBGAiterations) {
    int numKeys = numParameters / 2;
    int totalSize = numKeys * PRBGAiterations * 6;

    CHECK_CUDA(cudaMalloc((void**)d_keysAndControlPs,      numParameters * sizeof(double)), "cudaMalloc keysAndControlPs");
    CHECK_CUDA(cudaMalloc((void**)d_values4ByteStream_1,   totalSize * sizeof(unsigned char)), "cudaMalloc byteStream1");
    CHECK_CUDA(cudaMalloc((void**)d_values4ByteStream_2,   totalSize * sizeof(unsigned char)), "cudaMalloc byteStream2");
    CHECK_CUDA(cudaMalloc((void**)d_byteStreamFinal,      totalSize * sizeof(unsigned char)), "cudaMalloc byteStreamFinal");
}

static void device_frame_allocation_local(unsigned char ** d_sd_array, int numSubFrames) {
    CHECK_CUDA(cudaMalloc((void**)d_sd_array, numSubFrames * sizeof(unsigned char)), "cudaMalloc d_sd_array");
}

// =============================== MAIN ===============================
int main(int argc, char** argv) {

    std::string inputPath  = "testVideos/test_768x768_20fps.mp4";
    std::string outputPath = "testResults/encrypted_output.mp4";

    if (argc >= 2) inputPath  = argv[1];
    if (argc >= 3) outputPath = argv[2];

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open input video: " << inputPath << "\n";
        return -1;
    }

    // ✅ detect input resolution + fps
    gW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    gH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    double fps_d = cap.get(cv::CAP_PROP_FPS);
    gFPS = (int)std::round(fps_d);
    if (gFPS <= 0) gFPS = 20; // fallback

    if (gW <= 0 || gH <= 0) {
        std::cerr << "ERROR: invalid input size\n";
        return -1;
    }

    // ✅ you guaranteed this will always be true, keep guard anyway
    if ((gW % SUB_W) != 0 || (gH % SUB_H) != 0) {
        std::cerr << "ERROR: input size NOT multiple of subframe size\n";
        std::cerr << "Input = " << gW << "x" << gH
                  << " | subframe = " << SUB_W << "x" << SUB_H << "\n";
        return -1;
    }

    std::cout << "[main] input: " << gW << "x" << gH << " @ " << gFPS
              << " FPS | subframe " << SUB_W << "x" << SUB_H << "\n";

    // ✅ Jetson HW encode pipeline (same resolution as input)
    std::string pipeline =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(gW) +
        ",height=" + std::to_string(gH) +
        ",framerate=" + std::to_string(gFPS) + "/1 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=" + std::to_string(gW) +
        ",height=" + std::to_string(gH) +
        ",framerate=" + std::to_string(gFPS) + "/1 ! "
        "nvv4l2h264enc bitrate=8000000 insert-sps-pps=true iframeinterval=30 ! "
        "h264parse ! qtmux ! filesink location=" + outputPath + " sync=false";

    cv::VideoWriter writer(pipeline, cv::CAP_GSTREAMER, 0, gFPS, cv::Size(gW, gH), true);

    if (!writer.isOpened()) {
        std::cerr << "ERROR: Could not open GStreamer VideoWriter\n";
        std::cerr << "TIP: OpenCV must be built with GStreamer support.\n";
        return -1;
    }

    std::thread t1(capture_thread, std::ref(cap));
    std::thread t2(encrypt_thread);
    std::thread t3(writer_thread, std::ref(writer));

    t1.join();
    t2.join();
    t3.join();

    std::cout << "[main] finished cleanly ✅\n";
    return 0;
}

// =============================== CAPTURE ===============================
void capture_thread(cv::VideoCapture& cap) {
    std::cout << "[capture] started\n";
    int id = 0;
    cv::Mat frame;

    while (cap.read(frame)) {

        // ✅ ensure size stays constant
        if (frame.cols != gW || frame.rows != gH) {
            std::cerr << "[capture] ERROR: frame size changed mid-stream: "
                      << frame.cols << "x" << frame.rows
                      << " expected " << gW << "x" << gH << "\n";
            break;
        }

        // ✅ force BGR 8UC3
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }
        if (frame.type() != CV_8UC3) {
            frame = frame.clone();
        }
        if (!frame.isContinuous()) frame = frame.clone();

        std::unique_lock<std::mutex> lk(m_in);
        cv_in_space.wait(lk, [] { return (int)q_in.size() < MAX_QUEUE_IN; });

        q_in.push({ frame.clone(), id++ });

        lk.unlock();
        cv_in.notify_one();
    }

    capture_done = true;
    cv_in.notify_all();
    std::cout << "[capture] done\n";
}

// =============================== ENCRYPT ===============================
void encrypt_thread() {
    std::cout << "[encrypt] started\n";

    const int width  = gW;
    const int height = gH;

    const int subframeWidth  = SUB_W;
    const int subframeHeight = SUB_H;

    const int PRBGAsAmount = (width * height) / 6;

    // ✅ PRBGAiterations cannot be 0
    int PRBGAiterations = (int)std::ceil(((float)(subframeHeight * subframeWidth * NUM_CHANNELS) / 6.0f)) / 16;
    PRBGAiterations = std::max(1, PRBGAiterations);

    const int numSubFrames = (width * height) / (subframeHeight * subframeWidth);

    const double globalKey = 0.223456;
    const double p = 0.3;

    std::vector<unsigned char> byteStreamFinal;

    unsigned char *d_values4ByteStream_1 = nullptr;
    unsigned char *d_values4ByteStream_2 = nullptr;
    unsigned char *d_byteStreamFinal     = nullptr;

    unsigned char *d_sd_array = nullptr;

    unsigned char *d_input[2]  = {nullptr, nullptr};
    unsigned char *d_output[2] = {nullptr, nullptr};

    double *d_keysAndControlPs = nullptr;

    double *keysAndControlPsPinned = nullptr;
    unsigned char *h_frame_pinned[2] = {nullptr, nullptr};

    int buffer_id[2] = {-1, -1};

    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1), "cudaStreamCreate");

    cudaEvent_t frameReady[2];
    CHECK_CUDA(cudaEventCreateWithFlags(&frameReady[0], cudaEventDisableTiming), "cudaEventCreate 0");
    CHECK_CUDA(cudaEventCreateWithFlags(&frameReady[1], cudaEventDisableTiming), "cudaEventCreate 1");

    const size_t frameBytes = (size_t)width * height * NUM_CHANNELS * sizeof(unsigned char);

    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMallocHost(&h_frame_pinned[i], frameBytes), "cudaMallocHost frame");
        CHECK_CUDA(cudaMalloc(&d_input[i], frameBytes), "cudaMalloc d_input");
        CHECK_CUDA(cudaMalloc(&d_output[i], frameBytes), "cudaMalloc d_output");
    }

    CHECK_CUDA(cudaMallocHost(&keysAndControlPsPinned, (2 * PRBGAsAmount + 1) * sizeof(double)), "cudaMallocHost keys");

    PRBGA_init_local(2 * PRBGAsAmount,
                     &d_keysAndControlPs,
                     &d_values4ByteStream_1,
                     &d_values4ByteStream_2,
                     &d_byteStreamFinal,
                     PRBGAiterations);

    device_frame_allocation_local(&d_sd_array, numSubFrames);

    int frameCount = 0;
    uint64_t sc = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    while (true) {
        FramePacket pkt;

        {
            std::unique_lock<std::mutex> lk(m_in);
            cv_in.wait(lk, [] { return !q_in.empty() || capture_done.load(); });

            if (q_in.empty() && capture_done.load()) break;

            pkt = q_in.front();
            q_in.pop();
        }
        cv_in_space.notify_one();

        int cur  = frameCount % 2;
        int prev = (frameCount + 1) % 2;

        buffer_id[cur] = pkt.id;

        generatePRBGMainKeysv3Pinned(globalKey, p, 2 * PRBGAsAmount + 1, &sc, keysAndControlPsPinned);

        std::memcpy(h_frame_pinned[cur], pkt.frame.data, frameBytes);

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

        int performInverseConfusion = 0;
        int performInverseDiffusion = 0;

        CHECK_CUDA(cudaMemcpyAsync(d_input[cur], h_frame_pinned[cur], frameBytes,
                                   cudaMemcpyHostToDevice, stream1),
                   "cudaMemcpyAsync H2D");

        for (int i = 0; i < ROUNDS; i++) {
            confusionOpWrapper(pkt.frame.data, d_input[cur], pkt.frame.data, d_output[cur],
                               width, height, subframeHeight, sc, performInverseConfusion, stream1);
            std::swap(d_input[cur], d_output[cur]);

            diffusionOpWrapper(d_byteStreamFinal, d_input[cur], pkt.frame.data, d_output[cur],
                               pkt.frame.data, width, height, subframeHeight, subframeWidth,
                               performInverseDiffusion, d_sd_array, stream1);
            std::swap(d_input[cur], d_output[cur]);
        }

        CHECK_CUDA(cudaMemcpyAsync(h_frame_pinned[cur], d_input[cur], frameBytes,
                                   cudaMemcpyDeviceToHost, stream1),
                   "cudaMemcpyAsync D2H");

        CHECK_CUDA(cudaEventRecord(frameReady[cur], stream1), "cudaEventRecord");

        if (frameCount >= 1) {
            CHECK_CUDA(cudaEventSynchronize(frameReady[prev]), "cudaEventSynchronize(prev)");

            cv::Mat out(height, width, CV_8UC3);
            std::memcpy(out.data, h_frame_pinned[prev], frameBytes);

            {
                std::unique_lock<std::mutex> lk(m_out);
                cv_out_space.wait(lk, [] { return (int)q_out.size() < MAX_QUEUE_OUT; });
                q_out.push({ out, buffer_id[prev] });
            }
            cv_out.notify_one();
        }

        frameCount++;

        if (frameCount % 30 == 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double seconds = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "encrypted frames: " << frameCount << " | FPS " << (frameCount / seconds) << "\n";
        }
    }

    // flush last in-flight frame
    if (frameCount > 0) {
        int last = (frameCount - 1) % 2;
        CHECK_CUDA(cudaEventSynchronize(frameReady[last]), "cudaEventSynchronize(last)");

        cv::Mat out(height, width, CV_8UC3);
        std::memcpy(out.data, h_frame_pinned[last], frameBytes);

        {
            std::unique_lock<std::mutex> lk(m_out);
            cv_out_space.wait(lk, [] { return (int)q_out.size() < MAX_QUEUE_OUT; });
            q_out.push({ out, buffer_id[last] });
        }
        cv_out.notify_one();
    }

    encrypt_done = true;
    cv_out.notify_all();

    CHECK_CUDA(cudaStreamSynchronize(stream1), "cudaStreamSynchronize");

    cudaStreamDestroy(stream1);
    cudaEventDestroy(frameReady[0]);
    cudaEventDestroy(frameReady[1]);

    cudaFree(d_byteStreamFinal);

    for (int i = 0; i < 2; i++) {
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaFreeHost(h_frame_pinned[i]);
    }

    cudaFreeHost(keysAndControlPsPinned);

    cudaFree(d_keysAndControlPs);
    cudaFree(d_values4ByteStream_1);
    cudaFree(d_values4ByteStream_2);
    cudaFree(d_sd_array);

    std::cout << "[encrypt] done\n";
}

// =============================== WRITER ===============================
void writer_thread(cv::VideoWriter& writer) {
    std::cout << "[writer] started\n";

    while (true) {
        FramePacket pkt;

        {
            std::unique_lock<std::mutex> lk(m_out);
            cv_out.wait(lk, [] { return !q_out.empty() || encrypt_done.load(); });

            if (q_out.empty() && encrypt_done.load()) break;

            pkt = q_out.front();
            q_out.pop();
        }

        cv_out_space.notify_one();
        writer.write(pkt.frame);
    }

    std::cout << "[writer] done\n";
}
