#ifndef ENCRYPT_KERNEL_HPP
#define ENCRYPT_KERNEL_HPP
#include <stdint.h>
#include <vector>

//host-side wrapper function which is called from main.cpp

void encryptFrame(uint8_t *input, uint8_t * output, int width, int height) ;
void PRBGAandByteStreamGenWrapper(const std::vector<double>& keysAndControlPs, std::vector<uint8_t>& byteStreamFinal, std::vector<uint8_t>& output1_bytes, std::vector<uint8_t>& output2_bytes, const int PRBGAiterations);

#endif