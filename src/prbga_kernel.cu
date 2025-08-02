#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "../include/encrypt_kernel.hpp"

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

__global__ void prbgaKernel( double *output1, double *output2, const double *prbga_keys_and_control_ps, double sc, int numValuesGeneratedPRBGA) {

    int blockId = blockIdx.x ;

    //only one thread per block runs the PRBGa... for now
    if(threadIdx.x == 0) {

        // we are multiplying by 0.5 because x0 and p input to the PRBGA need to be in range (0, 0.5) as mentioned by research paper...
        //...while right now they are in range [0,1] as the xi generic outputs of the PRBG main are spit in range (0,1).. (always mentioned in the article)
        
        double x = 0.5 * prbga_keys_and_control_ps[2*blockId] ; // each PRBGa starts from a different key, which is the x0 of the PLCM
        double k = 0.5 * prbga_keys_and_control_ps[2*blockId+1] ; //the neighbor to the x0 is the control parameter p for the general PRBGA

        double *myOutput1 = output1 + blockId * numValuesGeneratedPRBGA ;
        double *myOutput2 = output2 + blockId * numValuesGeneratedPRBGA ;

        prbga_plcm( x, k,  myOutput1, numValuesGeneratedPRBGA ) ;
        prbga_plcm( x, k*0.5, myOutput2, numValuesGeneratedPRBGA) ;
        //prbg_results[idx] = x * sc :
    }
}

//the wrapper for the above kernel 

void PRBGAKernelWrapper(const std::vector<double>& keysAndControlPs, std::vector<double>& results, double sc, int PRBGAiterations){

    int numParameters = keysAndControlPs.size() - 1; //-1 because we have taken sc out of that vector
    int numKeys = numParameters/2;

    int totalSize = numKeys * PRBGAiterations ;//should be 128 * 768 . is the total size which considers all the arrays of values reproduced by all called PRBGAs

    //allocating device memory
    double * d_keysAndControlPs; // creating cuda array pointer for containing keys and control parameter inputs to the various PRBGAs

    double * d_values4ByteStream_1 ; 
    double * d_values4ByteStream_2 ;

    cudaMalloc(&d_keysAndControlPs, numParameters * sizeof( double ) );

    //allocating space for each individual array on the device
    cudaMalloc( (void **) &d_values4ByteStream_1, totalSize * sizeof( double ) );
    cudaMalloc( (void **) &d_values4ByteStream_2, totalSize * sizeof( double ) );

    //copy keys and controlParameters to device 
    cudaMemcpy(d_keysAndControlPs, keysAndControlPs.data(), numParameters * sizeof(double), cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 blocks(numKeys);
    dim3 threads(1);

    prbgaKernel<<<blocks, threads>>>( d_values4ByteStream_1, d_values4ByteStream_2,
                                     d_keysAndControlPs, sc, PRBGAiterations ) ;
    cudaDeviceSynchronize();
   
    //copy results back  .... this is probably wrong, go back to it
    results.resize(2 * totalSize ) ;
    cudaMemcpy(results.data(), d_values4ByteStream_1, totalSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(results.data() + totalSize, d_values4ByteStream_2, totalSize * sizeof(double), cudaMemcpyDeviceToHost) ;
    
    //clean up
    cudaFree(d_keysAndControlPs);
    cudaFree(d_values4ByteStream_1);
    cudaFree(d_values4ByteStream_2);

}