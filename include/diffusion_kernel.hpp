#ifndef DIFFUSION_KERNEL_HPP
#define DIFFUSION_KERNEL_HPP

#include <vector>
#include <opencv2/opencv.hpp> 
#include <cstdint>

void diffusionOpWrapper( unsigned char *d_byteStream, 
                         unsigned char *d_input, 
                         unsigned char *input, 
                         unsigned char *d_output, 
                         unsigned char *output,  
                         int width, 
                         int height, 
                         int subframe_height, 
                         int subframe_width, 
                         int performInverseDiffusion, 
                         unsigned char *d_sd_array,
                         cudaStream_t stream1 ) ;

void runCoalescedLoadWrapper( unsigned char *input, unsigned char *output , int width, int height ) ;

#endif 