#ifndef ENCRYPT_KERNEL_HPP
#define ENCRYPT_KERNEL_HPP
#include <stdint.h>

//host-side wrapper function which is called from main.cpp

void encryptFrame(uint8_t *input, uint8_t * output, int width, int height) ;

#endif