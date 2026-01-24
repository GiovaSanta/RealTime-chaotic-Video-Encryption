#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Tuned so copies and kernel are comparable
constexpr int WIDTH  = 2048;
constexpr int HEIGHT = 2048;
constexpr size_t BYTES = WIDTH * HEIGHT;

__global__ void fakeKernel(uint8_t* in, uint8_t* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BYTES) return;

    volatile int x = idx;
    for (int i = 0; i < 120; i++) {   // SHORT kernel
        x = x * 17 + 11;
    }

    out[idx] = in[idx] ^ (x & 0xFF);
}

int main()
{
    constexpr int NUM_BUFFERS = 2;
    constexpr int NUM_FRAMES  = 30;

    uint8_t* h_in [NUM_BUFFERS];
    uint8_t* h_out[NUM_BUFFERS];
    uint8_t* d_in [NUM_BUFFERS];
    uint8_t* d_out[NUM_BUFFERS];
    cudaStream_t streams[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK(cudaMallocHost(&h_in[i],  BYTES));
        CHECK(cudaMallocHost(&h_out[i], BYTES));
        CHECK(cudaMalloc(&d_in[i],  BYTES));
        CHECK(cudaMalloc(&d_out[i], BYTES));
        CHECK(cudaStreamCreate(&streams[i]));
        memset(h_in[i], i, BYTES);
    }

    dim3 block(256);
    dim3 grid((BYTES + block.x - 1) / block.x);

    printf("Starting balanced kernel/copy pipeline demo...\n");

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int s = frame % NUM_BUFFERS;

        // First H→D copy
        CHECK(cudaMemcpyAsync(
            d_in[s], h_in[s], BYTES,
            cudaMemcpyHostToDevice, streams[s]
        ));

        // Second H→D copy (amplify copy time)
        CHECK(cudaMemcpyAsync(
            d_out[s], h_in[s], BYTES,
            cudaMemcpyHostToDevice, streams[s]
        ));

        // Kernel (short)
        fakeKernel<<<grid, block, 0, streams[s]>>>(
            d_in[s], d_out[s]
        );

        // D→H copy
        CHECK(cudaMemcpyAsync(
            h_out[s], d_out[s], BYTES,
            cudaMemcpyDeviceToHost, streams[s]
        ));
    }

    CHECK(cudaDeviceSynchronize());
    printf("Pipeline completed.\n");

    for (int i = 0; i < NUM_BUFFERS; i++) {
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaFreeHost(h_in[i]);
        cudaFreeHost(h_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
