#ifndef DIFFUSION_KERNEL_HPP
#define DIFFUSION_KERNEL_HPP

#include <vector>
#include <opencv2/opencv.hpp> 
#include <cstdint>

void diffusionOpWrapper( unsigned char *input, unsigned char *output, unsigned char *byteStreamFinal, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, std::vector<unsigned char>& sd_array) ;
void runCoalescedLoadWrapper( unsigned char *input, unsigned char *output , int width, int height ) ;

#endif 