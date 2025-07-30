#include "encrypt_kernel.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void encryptFrameKernel(uint8_t *input, uint8_t *output, int width, int height) {

    //For now nothing is in this kernel. is dummmy kernl to check if compilation of main.cpp is succesull 

    int x = threadIdx.x + blockIdx.x * blockDim.x ;
    int y = threadIdx.y + blockIdx.y * blockDim.y ;
    
    int idx = y * width +x ;

    if( x < width && y < height ) {
        output[idx] = input[idx] - 50; //simple trasformation for now
    }

}

//Wrapper function for the CUDA kernel function.

 void encryptFrame(uint8_t *input, uint8_t *output, int width, int height) {
    
    uint8_t *d_input, *d_output;
    size_t size = width * height * sizeof(uint8_t);

    //Allocation of the device memory 
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); //256 threads per block for now
    dim3 gridDim((width + 15) / 16, (height + 15) / 16); //Cover entire image

    encryptFrameKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

}