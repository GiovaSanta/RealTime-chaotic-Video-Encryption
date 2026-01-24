#ifndef ENCRYPT_KERNEL_HPP
#define ENCRYPT_KERNEL_HPP
#include <stdint.h>
#include <vector>


void PRBGAandByteStreamGenWrapper( double * d_keysAndControlPs, 
                                   //const std::vector<double>& keysAndControlPs, 
                                   double * keysAndControlPs,
                                   int keysAndControlPsSize,
                                   unsigned char * d_values4ByteStream_1, 
                                   unsigned char * d_values4ByteStream_2, 
                                   unsigned char * d_byteStreamFinal, 
                                   std::vector<unsigned char>& byteStreamFinal, 
                                   const int PRBGAiterations, 
                                   int subframeHeight, 
                                   int subframeWidth, 
                                   int width, 
                                   int height,
                                   cudaStream_t stream1 );

#endif