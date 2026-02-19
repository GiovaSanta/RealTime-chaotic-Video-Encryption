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

#define ROUNDS 5
#define NUM_CHANNELS 3

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

// -------------------- Threads --------------------
void capture_thread(cv::VideoCapture& cap);
void encrypt_thread();
void writer_enc_thread(cv::VideoWriter& writer);
void writer_dec_thread(cv::VideoWriter& writer);

// =====================================================
//                      MAIN
// =====================================================
int main() {
    const int width  = 768;
    const int height = 768;
    const int fps    = 20;

    //cv::VideoCapture cap("testVideos/test_960x960_20fps.mp4");
    cv::VideoCapture cap("testVideos/test_768x768_20fps.mp4");
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open input video\n";
        return -1;
    }

    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Total frames = " << totalFrames << std::endl;

    // ---- ENCRYPTED OUTPUT (NVENC) ----
    std::string pipeline_enc =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=960,height=960,framerate=20/1 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=960,height=960,framerate=20/1 ! "
        "nvv4l2h264enc bitrate=8000000 insert-sps-pps=true iframeinterval=30 ! "
        "h264parse ! qtmux ! filesink location=testResults/encrypted_output.mp4 sync=false";

    cv::VideoWriter writer_enc(pipeline_enc, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);
    if (!writer_enc.isOpened()) {
        std::cerr << "ERROR: Could not open GStreamer VideoWriter (encrypted)\n";
        return -1;
    }

    // ---- DECRYPTED OUTPUT (NVENC) ----
    std::string pipeline_dec =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=960,height=960,framerate=20/1 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=960,height=960,framerate=20/1 ! "
        "nvv4l2h264enc bitrate=8000000 insert-sps-pps=true iframeinterval=30 ! "
        "h264parse ! qtmux ! filesink location=testResults/decrypted_check.mp4 sync=false";

    cv::VideoWriter writer_dec(pipeline_dec, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);
    if (!writer_dec.isOpened()) {
        std::cerr << "ERROR: Could not open GStreamer VideoWriter (decrypted)\n";
        return -1;
    }

    std::thread t1(capture_thread, std::ref(cap));
    std::thread t2(encrypt_thread);
    std::thread t3(writer_enc_thread, std::ref(writer_enc));
    std::thread t4(writer_dec_thread, std::ref(writer_dec));

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    std::cout << "[main] finished cleanly ✅\n";
    return 0;
}

// =====================================================
//                      CAPTURE
// =====================================================
void capture_thread(cv::VideoCapture& cap) {
    std::cout << "[capture] started\n";

    int id = 0;
    cv::Mat frame;

    while ( cap.read(frame) ) {

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

// =====================================================
//                   ENCRYPT + DECRYPT
// =====================================================

void encrypt_thread() {
    std::cout << "[encrypt] started\n";

    const int width  = 768;
    const int height = 768;

    const int subframeWidth  = 12;
    const int subframeHeight = 8;

    const int PRBGAsAmount = (width * height) / 6;

    const int PRBGAiterations =
        (int)std::ceil(((float)(subframeHeight * subframeWidth * NUM_CHANNELS) / 6.0f)) / 16;

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

    cudaEvent_t encReady[2];
    CHECK_CUDA(cudaEventCreateWithFlags(&encReady[0], cudaEventDisableTiming), "cudaEventCreate encReady0");
    CHECK_CUDA(cudaEventCreateWithFlags(&encReady[1], cudaEventDisableTiming), "cudaEventCreate encReady1");

    // allocate buffers
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMallocHost(&h_enc_pinned[i], frameBytes), "cudaMallocHost h_enc_pinned");
        CHECK_CUDA(cudaMallocHost(&h_dec_pinned[i], frameBytes), "cudaMallocHost h_dec_pinned");

        CHECK_CUDA(cudaMalloc(&d_input[i],  frameBytes), "cudaMalloc d_input");
        CHECK_CUDA(cudaMalloc(&d_output[i], frameBytes), "cudaMalloc d_output");
    }

    CHECK_CUDA(cudaMallocHost(&keysAndControlPsPinned, (2 * PRBGAsAmount + 1) * sizeof(double)),
               "cudaMallocHost keysAndControlPsPinned");

    PRBGA_init(2 * PRBGAsAmount,
               &d_keysAndControlPs,
               &d_values4ByteStream_1,
               &d_values4ByteStream_2,
               &d_byteStreamFinal,
               PRBGAiterations);

    device_frame_allocation(width * height, &d_sd_array, numSubFrames);

    int frameCount = 0;
    uint64_t sc = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    while (true) {
        FramePacket pkt;

        // ---- pop input ----
        {
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
        generatePRBGMainKeysv3Pinned(globalKey, p, 2 * PRBGAsAmount + 1, &sc, keysAndControlPsPinned);

        // ---- copy input frame -> pinned buffer ----
        std::memcpy(h_enc_pinned[cur], pkt.frame.data, frameBytes);

        // ---- PRBGA + bytestream generation ----
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

        // ---- H2D ----
        CHECK_CUDA(cudaMemcpyAsync(d_input[cur], h_enc_pinned[cur], frameBytes,
                                   cudaMemcpyHostToDevice, stream1),
                   "cudaMemcpyAsync H2D");

        // ===================== FORWARD ENCRYPT =====================
        int invConf = 0;
        int invDiff = 0;

        for (int r = 0; r < ROUNDS; r++) {
            confusionOpWrapper(pkt.frame.data, d_input[cur], pkt.frame.data, d_output[cur],
                               width, height, subframeHeight, sc, invConf, stream1);
            std::swap(d_input[cur], d_output[cur]);

            diffusionOpWrapper(d_byteStreamFinal, d_input[cur], pkt.frame.data, d_output[cur],
                               pkt.frame.data, width, height, subframeHeight, subframeWidth,
                               invDiff, d_sd_array, stream1);
            std::swap(d_input[cur], d_output[cur]);
        }

        // ---- D2H encrypted ----
        CHECK_CUDA(cudaMemcpyAsync(h_enc_pinned[cur], d_input[cur], frameBytes,
                                   cudaMemcpyDeviceToHost, stream1),
                   "cudaMemcpyAsync D2H(enc)");

        CHECK_CUDA(cudaEventRecord(encReady[cur], stream1), "cudaEventRecord(encReady)");

        // ---- push PREVIOUS encrypted frame (correct buffer!) ----
        if (frameCount >= 1) {
            CHECK_CUDA(cudaEventSynchronize(encReady[prev]), "cudaEventSync(prev)");

            cv::Mat out_enc(height, width, CV_8UC3);
            std::memcpy(out_enc.data, h_enc_pinned[prev], frameBytes);  // ✅ correct buffer

            {
                std::unique_lock<std::mutex> lk(m_enc);
                cv_enc_space.wait(lk, [] { return (int)q_out_enc.size() < MAX_QUEUE_OUT; });
                q_out_enc.push({ out_enc, buffer_id[prev] });
            }
            cv_enc.notify_one();
        }

        // ===================== INVERSE DECRYPT CHECK =====================
        // Now d_input[cur] currently holds the encrypted frame on GPU.
        // We invert rounds (DIFFUSION inverse first, then CONFUSION inverse).
        invConf = 1;
        invDiff = 1;

        for (int r = ROUNDS - 1; r >= 0; r--) {
            diffusionOpWrapper(d_byteStreamFinal, d_input[cur], pkt.frame.data, d_output[cur],
                               pkt.frame.data, width, height, subframeHeight, subframeWidth,
                               invDiff, d_sd_array, stream1);
            std::swap(d_input[cur], d_output[cur]);

            confusionOpWrapper(pkt.frame.data, d_input[cur], pkt.frame.data, d_output[cur],
                               width, height, subframeHeight, sc, invConf, stream1);
            std::swap(d_input[cur], d_output[cur]);
        }

        // ---- D2H decrypted ----
        CHECK_CUDA(cudaMemcpyAsync(h_dec_pinned[cur], d_input[cur], frameBytes,
                                   cudaMemcpyDeviceToHost, stream1),
                   "cudaMemcpyAsync D2H(dec)");

        // for correctness checking, we sync here
        CHECK_CUDA(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(after dec)");

        // ---- push decrypted frame ----
        {
            cv::Mat out_dec(height, width, CV_8UC3);
            std::memcpy(out_dec.data, h_dec_pinned[cur], frameBytes);

            std::unique_lock<std::mutex> lk(m_dec);
            cv_dec_space.wait(lk, [] { return (int)q_out_dec.size() < MAX_QUEUE_OUT; });
            q_out_dec.push({ out_dec, pkt.id });
        }
        cv_dec.notify_one();

        frameCount++;

        // ---- FPS print (encryption rate) ----
        if (frameCount % 30 == 0) {
            auto t1 = std::chrono::high_resolution_clock::now();
            double seconds = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "encrypted frames: " << frameCount << " | FPS " << (frameCount / seconds) << "\n";
        }
    }

    // ---- flush LAST encrypted frame (very important) ----
    if (frameCount > 0) {
        int last = (frameCount - 1) % 2;
        CHECK_CUDA(cudaEventSynchronize(encReady[last]), "cudaEventSync(last)");

        cv::Mat out_enc(height, width, CV_8UC3);
        std::memcpy(out_enc.data, h_enc_pinned[last], frameBytes);

        {
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
    cudaStreamDestroy(stream1);
    cudaEventDestroy(encReady[0]);
    cudaEventDestroy(encReady[1]);

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

    std::cout << "[encrypt] done\n";
}

// =====================================================
//                WRITER (ENCRYPTED)
// =====================================================
void writer_enc_thread(cv::VideoWriter& writer) {
    std::cout << "[writer_enc] started\n";

    while (true) {
        FramePacket pkt;

        {
            std::unique_lock<std::mutex> lk(m_enc);
            cv_enc.wait(lk, [] { return !q_out_enc.empty() || encrypt_done.load(); });

            if (q_out_enc.empty() && encrypt_done.load())
                break;

            pkt = q_out_enc.front();
            q_out_enc.pop();
        }
        cv_enc_space.notify_one();

        writer.write(pkt.frame);
    }

    std::cout << "[writer_enc] done\n";
}

// =====================================================
//                WRITER (DECRYPTED)
// =====================================================
void writer_dec_thread(cv::VideoWriter& writer) {
    std::cout << "[writer_dec] started\n";

    while (true) {
        FramePacket pkt;

        {
            std::unique_lock<std::mutex> lk(m_dec);
            cv_dec.wait(lk, [] { return !q_out_dec.empty() || encrypt_done.load(); });

            if (q_out_dec.empty() && encrypt_done.load())
                break;

            pkt = q_out_dec.front();
            q_out_dec.pop();
        }
        cv_dec_space.notify_one();

        writer.write(pkt.frame);
    }

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
