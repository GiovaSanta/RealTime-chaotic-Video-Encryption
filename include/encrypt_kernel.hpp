#ifndef ENCRYPT_KERNEL_HPP
#define ENCRYPT_KERNEL_HPP
#include <stdint.h>
#include <vector>

//host-side wrapper function which is called from main.cpp

void encryptFrame( uint8_t *input, 
                   uint8_t * output, 
                   int width, 
                   int height ) ;


void PRBGAandByteStreamGenWrapper( double * d_keysAndControlPs, 
                                   const std::vector<double>& keysAndControlPs, 
                                   unsigned char * d_values4ByteStream_1, 
                                   unsigned char * d_values4ByteStream_2, 
                                   unsigned char * d_byteStreamFinal, 
                                   std::vector<unsigned char>& byteStreamFinal, 
                                   std::vector<unsigned char>& output1_bytes, 
                                   std::vector<unsigned char>& output2_bytes, 
                                   const int PRBGAiterations, 
                                   int subframeHeight, 
                                   int subframeWidth, 
                                   int width, 
                                   int height );

#endif