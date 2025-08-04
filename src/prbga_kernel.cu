#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "../include/encrypt_kernel.hpp"

#define VALUESGENERATEDPRBGA 768 

//will need to redo this one function do it well following how the prbg for the main thread was implemented
__device__ void prbga_plcm(double xi, double p, double *output, int numValuesGeneratedPRBGA ){
    
    for ( int i = 0 ; i < numValuesGeneratedPRBGA ; i++ ) {
        if ( xi >= 0 && xi < p ) {
            xi = xi / p ;
        } 
        else if( xi >= p && xi <= 0.5 ){
            xi = (xi - p) / (0.5 - p) ;
        }
        else if( xi > 0.5 && xi <= 1.0 ){
            //originally in the paper there is a recursive call but as that recursion is always one stepped lets say... 
            //...so we can just avoid the recursion and instead write the checks again that the onestep recursion would have done
            xi = xi > 1.0 ? 1.0 : xi ; //we are clamping the value to 1 just in case
            xi = xi < 0.0 ? 0.0 : xi ; //similar if ever the value is negative
            xi = 1.0 - xi ;
            if ( xi >= 0 && xi < p){
                xi = xi / p;
            }
            else if ( xi >= p && xi <= 0.5){
                xi = (xi - p) / ( 0.5 - p );
            }
        }
        output[i]=xi ;
    }
}

__global__ void prbgaKernel( uint8_t *output1_bytes, uint8_t *output2_bytes, const double *prbga_keys_and_control_ps, double sc, const int numValuesGeneratedPRBGA) {

    int blockId = blockIdx.x ;
    int tid = threadIdx.x;

    __shared__ double PRBGA_values1[VALUESGENERATEDPRBGA];
    __shared__ double PRBGA_values2[VALUESGENERATEDPRBGA];

    //only one thread per block runs the PRBGa... for now
    if(tid == 0) {

        // we are multiplying by 0.5 because x0 and p input to the PRBGA need to be in range (0, 0.5) as mentioned by research paper...
        //...while right now they are in range [0,1] as the xi generic outputs of the PRBG main are spit in range (0,1).. (always mentioned in the article)
        
        double x = 0.5 * prbga_keys_and_control_ps[2*blockId] ; // each PRBGa starts from a different key, which is the x0 of the PLCM
        double k = 0.5 * prbga_keys_and_control_ps[2*blockId+1] ; //the neighbor to the x0 is the control parameter p for the general PRBGA

        //double *myOutput1 = output1 + blockId * numValuesGeneratedPRBGA ;
        //double *myOutput2 = output2 + blockId * numValuesGeneratedPRBGA ;

        prbga_plcm( x, k,  PRBGA_values1, numValuesGeneratedPRBGA ) ;
        prbga_plcm( x, k*0.5, PRBGA_values2, numValuesGeneratedPRBGA) ;
        //prbg_results[idx] = x * sc :
    }

    // Wait for thread 0 to finish PRBGA
    __syncthreads();

    // All threads now work in parallel on extraction
    if( tid < numValuesGeneratedPRBGA ){//768
        uint64_t bits1 = *reinterpret_cast<uint64_t*>(&PRBGA_values1[tid]);
        uint64_t bits2 = *reinterpret_cast<uint64_t*>(&PRBGA_values2[tid]);

        uint64_t mantissa1 = bits1 & 0x000FFFFFFFFFFFFF ;
        uint64_t mantissa2 = bits2 & 0x000FFFFFFFFFFFFF ;

        int byteIndex = (blockId * numValuesGeneratedPRBGA + tid) * 6;

        for (int j = 0; j < 6; ++j) {
            output1_bytes[byteIndex + j] = (mantissa1 >> (8 * (5 - j))) & 0xFF;
            output2_bytes[byteIndex + j] = (mantissa2 >> (8 * (5 - j))) & 0xFF;
        }
    } 
}

//the wrapper for the above kernel 

void PRBGAKernelWrapper(const std::vector<double>& keysAndControlPs, std::vector<uint8_t>& output1_bytes, std::vector<uint8_t>& output2_bytes, double sc, const int PRBGAiterations){

    int numParameters = keysAndControlPs.size() - 1; //-1 because we have taken sc out of that vector
    int numKeys = numParameters/2;

    int totalSize = numKeys * PRBGAiterations * 6 ;//should be 128 * 768 *8 . is the total size which considers all the arrays of values reproduced by all called PRBGAs. its expressed in bytes. 

    //allocating device memory
    double * d_keysAndControlPs; // creating cuda array pointer for containing keys and control parameter inputs to the various PRBGAs

    uint8_t * d_values4ByteStream_1 ; 
    uint8_t * d_values4ByteStream_2 ;

    cudaMalloc(&d_keysAndControlPs, numParameters * sizeof( double ) );

    //copy keys and controlParameters to device 
    cudaMemcpy(d_keysAndControlPs, keysAndControlPs.data(), numParameters * sizeof(double), cudaMemcpyHostToDevice);
    
    //allocating space for each individual array on the device
    cudaMalloc( (void **) &d_values4ByteStream_1, totalSize * sizeof( uint8_t ) );
    cudaMalloc( (void **) &d_values4ByteStream_2, totalSize * sizeof( uint8_t ) );

    //launch the kernel
    dim3 blocks(numKeys);
    dim3 threads(PRBGAiterations); //768 the threads per block are 768

    prbgaKernel<<<blocks, threads>>>( d_values4ByteStream_1, d_values4ByteStream_2,
                                     d_keysAndControlPs, sc, PRBGAiterations ) ;
    cudaDeviceSynchronize();
   
    //copy results back  
    output1_bytes.resize( totalSize ) ; //the total size is in bytes so we should be good
    output2_bytes.resize( totalSize ) ; 

    cudaMemcpy(output1_bytes.data(), d_values4ByteStream_1, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(output2_bytes.data(), d_values4ByteStream_2, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost) ;
    
    //clean up
    cudaFree(d_keysAndControlPs);
    cudaFree(d_values4ByteStream_1);
    cudaFree(d_values4ByteStream_2);

}