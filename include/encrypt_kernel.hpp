#ifndef ENCRYPT_KERNEL_HPP
#define ENCRYPT_KERNEL_HPP
#include <stdint.h>
#include <vector>

//host-side wrapper function which is called from main.cpp

void encryptFrame(uint8_t *input, uint8_t * output, int width, int height) ;
void PRBGAKernelWrapper(const std::vector<double>& keysAndControlPs, std::vector<double>& results, double sc, int iterations);

#endif