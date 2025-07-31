#include "../include/encrypt_kernel.hpp"
#include <cuda_runtime.h>
#include <iostream>

__device__ float prbg_PLCM(float x. float k){
    if(x >= 0.0f && x < k)
        return x/k ;
    else if ( x >= k && x < 0.5f)
        return (x - k) / (0.5f - k) ;
    else if ( x >= 0.5f && x< 1.0f - k)
        return (1.0f - k - x) / (0.5f - k);
    else
        return (1.0f - x) / k ;
}

__global__ void encryptFrameKernel(const unsigned char *input, unsigned char *output, int width, int height, float seed, float k) {

    int x = threadIdx.x + blockIdx.x * blockDim.x ;
    int y = threadIdx.y + blockIdx.y * blockDim.y ;
    
    int idx = y * width + x ;

    if( x < width && y < height ) {
       
        //initializing chaotic value using a seed + per-thread offset
        float chaos = seed  ;
       
        //iterating the PLCM (Piecewise Linear Chaotic Map) for a few times
        for ( int i = 0 ; i < 5; ++i) {
            chaos = prbg_PLCM(chaos, k);
        }

        //creating a pseudoRandom byte
        unsigned char key = static_cast<unsigned char>(chaos * 255.0f);
        //XOR with pixel
        output[idx] = input[idx] ^ key;
    
    }

}

//Wrapper function for the CUDA kernel function.

 void encryptFrame(const cv::Mat& input, cv::Mat& output, float seed, float k) {
    
    int width = input.cols;
    int height = input.rows;
    unsigned char *d_input, *d_output;
    size_t size = width * height * sizeof(unsigned char);

    //Allocation of the device memory 
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); //256 threads per block for now
    dim3 gridDim((width + blockDim.x -1) / blockDim.x,
                 (height + blockDim.y -1) / blockDim.y); //Cover entire image

    encryptFrameKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, seed, k);

    //cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

}