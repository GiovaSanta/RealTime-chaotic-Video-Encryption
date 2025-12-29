#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "../include/encrypt_kernel.hpp"

//#define VALUESGENERATEDPRBGA 768 

//will need to redo this one function do it well following how the prbg for the main thread was implemented
__device__ void prbga_plcm(double xi, double p, double *output, int numValuesGeneratedPRBGA ){
    
    for ( int i = 0 ; i < numValuesGeneratedPRBGA ; i++ ) {
        if ( xi >= 0 && xi < p ) {
            xi = xi / p ;
        } 
        else if( xi >= p && xi <= 0.5 ) {
            xi = (xi - p) / (0.5 - p) ;
        }
        else if( xi > 0.5 && xi <= 1.0 ) {
            //originally in the paper there is a recursive call but as that recursion is always one stepped lets say... 
            //...so we can just avoid the recursion and instead write the checks again that the onestep recursion would have done
            //xi = xi > 1.0 ? 1.0 : xi ; //we are clamping the value to 1 just in case
            //xi = xi < 0.0 ? 0.0 : xi ; //similar if ever the value is negative
            xi = 1.0 - xi ;
            if ( xi >= 0 && xi < p) {
                xi = xi / p;
            }
            else if ( xi >= p && xi <= 0.5) {
                xi = (xi - p) / ( 0.5 - p );
            }
        }
        output[i]=xi ;
    }
}

//PRBGA iterations = 1;
__device__ void prbga_plcm_v3(double xi, double p, double *output, int numValuesGeneratedPRBGA, int index){
    
    if ( xi >= 0 && xi < p ) {
        xi = xi / p ;
    } 
    else if( xi >= p && xi <= 0.5 ) {
        xi = (xi - p) / (0.5 - p) ;
    }
    else if( xi > 0.5 && xi <= 1.0 ) {
        //originally in the paper there is a recursive call but as that recursion is always one stepped lets say... 
        //...so we can just avoid the recursion and instead write the checks again that the onestep recursion would have done
        //xi = xi > 1.0 ? 1.0 : xi ; //we are clamping the value to 1 just in case
        //xi = xi < 0.0 ? 0.0 : xi ; //similar if ever the value is negative
        xi = 1.0 - xi ;
        if ( xi >= 0 && xi < p) {
            xi = xi / p;
        }
        else if ( xi >= p && xi <= 0.5) {
            xi = (xi - p) / ( 0.5 - p );
        }
    }
        output[index]=xi ;
}

__device__ __forceinline__ void prbga_plcm_v2( double xi, const double p, double* __restrict__ output, int n) {
    
    const double inv_p = 1.0 / p;
    const double inv_half_minus_p = 1.0 / (0.5 - p);

    #pragma unroll 1
    for (int i = 0; i < n; ++i) {

        // Reflect if xi > 0.5
        if (xi > 0.5) {
            xi = 1.0 - xi;
        }

        // Main PLCM map
        if (xi < p) {
            xi *= inv_p;
        } else {
            xi = (xi - p) * inv_half_minus_p;
        }

        output[i] = xi;
    }
}

__global__ void prbgaKernel( unsigned char *finalByteStream, unsigned char *output1_bytes, unsigned char *output2_bytes, const double *prbga_keys_and_control_ps, const int numValuesGeneratedPRBGA ) {
 
    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;  
    int tid_x = threadIdx.x ;


    //__shared__ double PRBGA_values1[VALUESGENERATEDPRBGA];
    //__shared__ double PRBGA_values2[VALUESGENERATEDPRBGA]; 

    extern __shared__ double shared_mem[]; // the shared memory that is getting used per block reserved after callling the kernel will actually get destroyed automatically after the block is done.. so no need to free

    double *PRBGA_values1 = shared_mem ;
    double *PRBGA_values2 = shared_mem + numValuesGeneratedPRBGA ;

    //only one thread per block runs the PRBGa... for now
    if ( tid_x == 0 ) {

        //printf("Running PRBGA kernel in blockId_x= %d, blockId_y= %d\n", blockId_x, blockId_y);  

        // we are multiplying by 0.5 because x0 and p input to the PRBGA need to be in range (0, 0.5) as mentioned by research paper...
        //...while right now they are in range [0,1] as the xi generic outputs of the PRBG main are spit in range (0,1).. (always mentioned in the article)
        
        double x = 0.5 * prbga_keys_and_control_ps[ 2* (blockId_y * gridDim.x + blockId_x) ] ; // each PRBGa starts from a different key, which is the x0 of the PLCM
        double k = 0.5 * prbga_keys_and_control_ps[ 2* (blockId_y * gridDim.x + blockId_x) + 1 ] ; //the neighbor to the x0 is the control parameter p for the general PRBGA

        //double *myOutput1 = output1 + blockId * numValuesGeneratedPRBGA ;
        //double *myOutput2 = output2 + blockId * numValuesGeneratedPRBGA ;

        prbga_plcm( x, k, PRBGA_values1, numValuesGeneratedPRBGA ) ;
        prbga_plcm( x, k*0.5, PRBGA_values2, numValuesGeneratedPRBGA) ;

        //prbga_plcm_v2( x, k, PRBGA_values1, numValuesGeneratedPRBGA ) ;
        //prbga_plcm_v2( x, k*0.5, PRBGA_values2, numValuesGeneratedPRBGA) ;

        //if( blockId_x == 12 && blockId_y == 28 ){
            
            //printf("\ngridDim.x = %d\n", gridDim.x ) ;
            //printf("\nPRBGA corresponding to subframe(%d,%d) values\n", blockId_x, blockId_y) ;
            //for(int i = 0; i < 512; i ++){
            //    printf("%f ", PRBGA_values1[i]);
            //}

        //}
        //prbg_results[idx] = x * sc :
    
    }

    // Wait for thread 0 to finish PRBGA
    //__syncthreads();

    // All threads now work in parallel on extraction
    //if( tid_x < numValuesGeneratedPRBGA ) {  // tid_x is between 0 and 511
        uint64_t bits1 = *reinterpret_cast<uint64_t*>(&PRBGA_values1[tid_x]) ;
        uint64_t bits2 = *reinterpret_cast<uint64_t*>(&PRBGA_values2[tid_x]) ;

        uint64_t mantissa1 = bits1 & 0x000FFFFFFFFFFFFF ;
        uint64_t mantissa2 = bits2 & 0x000FFFFFFFFFFFFF ;

        int byteIndex = (blockId_y * gridDim.x * numValuesGeneratedPRBGA + numValuesGeneratedPRBGA * blockId_x + tid_x)*6 ;

        for (int j = 0; j < 6; ++j) {
            output1_bytes[byteIndex + j] = (mantissa1 >> (8 * (5 - j))) & 0xFF ;
            output2_bytes[byteIndex + j] = (mantissa2 >> (8 * (5 - j))) & 0xFF ;
            finalByteStream[byteIndex +j] = output1_bytes[byteIndex + j] ^ output2_bytes[byteIndex + j]; //the final byteStream derived from the previous calculated byte streams
        }
    //} 

}

__global__ void prbgaKernelv2( unsigned char *finalByteStream, unsigned char *output1_bytes, unsigned char *output2_bytes, const double *prbga_keys_and_control_ps, const int numValuesGeneratedPRBGA ) {
 
    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;  
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;

    //__shared__ double PRBGA_values1[VALUESGENERATEDPRBGA];
    //__shared__ double PRBGA_values2[VALUESGENERATEDPRBGA]; 

    extern __shared__ double shared_mem[]; // the shared memory that is getting used per block reserved after callling the kernel will actually get destroyed automatically after the block is done.. so no need to free

    double *PRBGA_values1 = shared_mem ;
    double *PRBGA_values2 = shared_mem + 16 ;

    //only one thread per block runs the PRBGa... for now
    //if( tid_x == 0 || tid_x == 6 ) {

        //printf("Running PRBGA kernel in blockId_x= %d, blockId_y= %d\n", blockId_x, blockId_y);  

        // we are multiplying by 0.5 because x0 and p input to the PRBGA need to be in range (0, 0.5) as mentioned by research paper...
        //...while right now they are in range [0,1] as the xi generic outputs of the PRBG main are spit in range (0,1).. (always mentioned in the article)
        
        //double x = 0.5 * prbga_keys_and_control_ps[ 2 * (blockId_y * gridDim.x + blockId_x) * tid_y ] ; // each PRBGa starts from a different key, which is the x0 of the PLCM
        //double k = 0.5 * prbga_keys_and_control_ps[ 2 * (blockId_y * gridDim.x + blockId_x) * tid_y + 1 ] ; //the neighbor to the x0 is the control parameter p for the general PRBGA

        double x = 0.5 * prbga_keys_and_control_ps[ 2 * (  (blockId_y * gridDim.x * blockDim.x * blockDim.y ) 
                                                        +  (blockId_x * blockDim.x ) 
                                                        +  (tid_y * gridDim.x * blockDim.x ) + tid_x ) ] ;

        double k = 0.5 * prbga_keys_and_control_ps[ 2 * (  (blockId_y * gridDim.x * blockDim.x * blockDim.y ) 
                                                        +  (blockId_x * blockDim.x ) 
                                                        +  (tid_y * gridDim.x * blockDim.x ) + tid_x ) + 1 ] ;

        //double *myOutput1 = output1 + blockId * numValuesGeneratedPRBGA ;
        //double *myOutput2 = output2 + blockId * numValuesGeneratedPRBGA ;

        prbga_plcm( x, k, PRBGA_values1, numValuesGeneratedPRBGA ) ;
        prbga_plcm( x, k*0.5, PRBGA_values2, numValuesGeneratedPRBGA) ;

        //if( blockId_x == 12 && blockId_y == 28 ){
            
            //printf("\ngridDim.x = %d\n", gridDim.x ) ;
            //printf("\nPRBGA corresponding to subframe(%d,%d) values\n", blockId_x, blockId_y) ;
            //for(int i = 0; i < 512; i ++){
            //    printf("%f ", PRBGA_values1[i]);
            //}

        //}
        //prbg_results[idx] = x * sc :
    
    //}

    // Wait for thread 0 to finish PRBGA
    //__syncthreads();

    // All threads now work in parallel on extraction
    //if( tid_x < numValuesGeneratedPRBGA ) {  // tid_x is between 0 and 511
        uint64_t bits1 = *reinterpret_cast<uint64_t*>( &PRBGA_values1[tid_y * blockDim.x + tid_x] ) ;
        uint64_t bits2 = *reinterpret_cast<uint64_t*>( &PRBGA_values2[tid_y * blockDim.x + tid_x] ) ;

        uint64_t mantissa1 = bits1 & 0x000FFFFFFFFFFFFF ;
        uint64_t mantissa2 = bits2 & 0x000FFFFFFFFFFFFF ;

        //int byteIndex = (blockId_y * gridDim.x * numValuesGeneratedPRBGA + numValuesGeneratedPRBGA * blockId_x + tid_x)*6 ;

        int byteIndex = (  (blockId_y * gridDim.x * blockDim.x * blockDim.y ) 
                        +  (blockId_x * blockDim.x ) 
                        +  (tid_y * gridDim.x * blockDim.x ) + tid_x ) ;

        for (int j = 0; j < 6; ++j) {
            output1_bytes[byteIndex + j] = (mantissa1 >> (8 * (5 - j))) & 0xFF ;
            output2_bytes[byteIndex + j] = (mantissa2 >> (8 * (5 - j))) & 0xFF ;
            finalByteStream[byteIndex +j] = output1_bytes[byteIndex + j] ^ output2_bytes[byteIndex + j]; //the final byteStream derived from the previous calculated byte streams
        }
    //} 

}

__global__ void prbgaKernelv3( unsigned char *finalByteStream, unsigned char *output1_bytes, unsigned char *output2_bytes, const double *prbga_keys_and_control_ps, const int numValuesGeneratedPRBGA ) {
 
    int blockId_x = blockIdx.x ;
    int tid_x = threadIdx.x ; 

    extern __shared__ double shared_mem[]; // the shared memory that is getting used per block reserved after callling the kernel will actually get destroyed automatically after the block is done.. so no need to free

    double *PRBGA_values1 = shared_mem ;
    double *PRBGA_values2 = shared_mem + blockDim.x ;

    double x = 0.5 * prbga_keys_and_control_ps[ 2 * ( blockId_x * blockDim.x + tid_x ) ] ;
    double k = 0.5 * prbga_keys_and_control_ps[ 2 * ( blockId_x * blockDim.x + tid_x ) + 1 ] ;

    /*if( blockId_x == 0 ) {
        //printf("x: %f\n", x);
    } */

    prbga_plcm_v3( x, k, PRBGA_values1, numValuesGeneratedPRBGA, tid_x ) ;
    prbga_plcm_v3( x, k*0.5, PRBGA_values2, numValuesGeneratedPRBGA, tid_x) ;

    /*if( blockId_x == 0 ) {
        printf("boh: %f\n", PRBGA_values1[tid_x]) ;
    }*/

    uint64_t bits1 = *reinterpret_cast<uint64_t*>( &PRBGA_values1[tid_x] ) ;
    uint64_t bits2 = *reinterpret_cast<uint64_t*>( &PRBGA_values2[tid_x] ) ;

    uint64_t mantissa1 = bits1 & 0x000FFFFFFFFFFFFF ;
    uint64_t mantissa2 = bits2 & 0x000FFFFFFFFFFFFF ;

    int byteIndex = ( blockId_x * blockDim.x + tid_x ) * 6 ;

        for (int j = 0; j < 6; ++j) {
            output1_bytes[byteIndex + j] = (mantissa1 >> (8 * (5 - j))) & 0xFF ;
            output2_bytes[byteIndex + j] = (mantissa2 >> (8 * (5 - j))) & 0xFF ;
            finalByteStream[byteIndex +j] = output1_bytes[byteIndex + j] ^ output2_bytes[byteIndex + j]; //the final byteStream derived from the previous calculated byte streams
        } 

}

void PRBGAandByteStreamGenWrapper(double * d_keysAndControlPs, 
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
                                  int height ) {

    int numParameters = keysAndControlPs.size(); // parameters being the keys and control parameters for the subfframes

    //printf("numParameters: %d\n", numParameters); //12800
    
    int numKeys = numParameters/2 ;
    //printf("numKeys: %d\n", numKeys) ;

    int blockdimx = ( width ) / ( subframeWidth ) ; 
    int blockdimy = ( height ) / ( subframeHeight ) ; 

    int totalSize = numKeys * PRBGAiterations * 6 ; //should be 128 * 768 *6 . is the total size which considers all the arrays of values reproduced by all called PRBGAs. its expressed in bytes. 

    //printf("total size finalbyteArray: %d\n", totalSize) ;

    //printf("singular PRBGa iterations:%d\n", PRBGAiterations) ;
    //printf("numParameters: %d\n", numParameters ) ;

    //printf("blockDim x : %d\n", blockdimx ) ;
    //printf("blockDim y : %d\n", blockdimy ) ;
    
    //unsigned char * d_values4ByteStream_1 ; 
    //unsigned char * d_values4ByteStream_2 ; 

    //unsigned char * d_byteStreamFinal ; // this is the final byte stream array containing all the 128 bytestream arrays contigously in memory. 
                                        // it is of size 128 * 768 * 6 bytes, and it will be used for the diffusion step of the encryption.

    //copy keys and controlParameters to device 
    cudaMemcpy(d_keysAndControlPs, keysAndControlPs.data(), numParameters * sizeof( double ), cudaMemcpyHostToDevice);
    
    //printf("%f\n", d_keysAndControlPs[0]);

    //launch the kernel
    //dim3 blocks( blockdimx, blockdimy ) ;  
    //dim3 threads( subframeWidth, subframeHeight*2 ) ; 
    //dim3 threads( 8, 2 ) ;
    //dim3 threads( PRBGAiterations ) ;

    dim3 threads( 64 ) ;
    dim3 blocks( 2400 ) ; //153600 prbgas / 512 

    //size_t sharedMemPerBlock = 2 * PRBGAiterations * sizeof(double) ;
    size_t sharedMemPerBlock = 2 * 64 * sizeof(double);

    //printf("prbga iterations: %d\n", PRBGAiterations );

    prbgaKernelv3<<<blocks, threads, sharedMemPerBlock>>>( d_byteStreamFinal, d_values4ByteStream_1, d_values4ByteStream_2, d_keysAndControlPs, PRBGAiterations ) ;

    //cudaDeviceSynchronize();
   
    //copy results back  
    output1_bytes.resize( totalSize ) ; //the total size is in bytes so we should be good
    output2_bytes.resize( totalSize ) ; 

    //printf("bytestream final size: %d\n", byteStreamFinal.size() ) ;
    byteStreamFinal.resize ( totalSize ) ;

    //cudaMemcpy(output1_bytes.data(), d_values4ByteStream_1, totalSize * sizeof( unsigned char), cudaMemcpyDeviceToHost );
    //cudaMemcpy(output2_bytes.data(), d_values4ByteStream_2, totalSize * sizeof( unsigned char), cudaMemcpyDeviceToHost ) ;
    cudaMemcpy(byteStreamFinal.data(), d_byteStreamFinal, totalSize * sizeof( unsigned char), cudaMemcpyDeviceToHost ) ;
    
    //clean up
    cudaFree(d_keysAndControlPs);
    cudaFree(d_values4ByteStream_1);
    cudaFree(d_values4ByteStream_2);
    cudaFree(d_byteStreamFinal);

}