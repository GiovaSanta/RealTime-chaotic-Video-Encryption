#ifndef PRBG_MAIN_PLCM_HPP
#define PRBG_MAIN_PLCM_HPP

#include <vector> 
#include <iostream>

std::vector<double> generatePRBGMainKeys(double x0, double p, int numParameters4subsequentPRBGas, uint64_t *sc) ;

std::vector<double> generatePRBGMainKeysv2( double x0, double p, int n, uint64_t *sc) ;

void generatePRBGMainKeysv3Pinned( double x0, double p, int n, uint64_t* sc, double* out ) ;

void PRBGA_init( int numParameters, 
                 double ** d_keysAndControlPs, 
                 unsigned char ** d_values4ByteStream_1, 
                 unsigned char **d_values4ByteStream_2, 
                 unsigned char ** d_byteStreamFinal, 
                 int PRBGAiterations ) ; //memory allocations for the arrays used in prbga kernel

#endif