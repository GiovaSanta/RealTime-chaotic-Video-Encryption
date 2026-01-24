#ifndef CONFUSION_KERNEL_HPP
#define CONFUSION_KERNEL_HPP

#include <vector>
#include <opencv2/opencv.hpp> 
#include <cstdint>

void confusionOpWrapper( unsigned char *input, 
                         unsigned char * d_input,
                         unsigned char *output,
                         unsigned char *d_output,
                         int width ,
                         int height,
                         int subframe_height,
                         uint64_t sc,
                         int performInverseConfusion,
                         cudaStream_t stream1 );

#endif 