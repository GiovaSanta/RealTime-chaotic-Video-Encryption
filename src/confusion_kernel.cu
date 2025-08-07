#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/confusion_kernel.hpp"
#include <math.h>
#include <iostream>


__global__ void inverseConfusionKernel ( const unsigned char *input, unsigned char *output, int width, uint64_t sc ){
    int tid_x = threadIdx.x ; //0 to 153
    int tid_y = threadIdx.y ; // 0 to 5
    
    int num_blocks = 128; // there are 128 blocks

    int startRow =   blockIdx.x * (width/num_blocks) ;
    //printf("startRow: %d", startRow) ;

    int x_base = tid_x * 5; // starting column index for this particular thread
    int y = startRow + tid_y; // actual row being processed  

    //if ( y >= startRow + 6 ) return ; 

    for(int i = 0; i< 5; i ++){
        int x = x_base + i ;
        if ( x < width ){  // maybe can remove this if with padding in th future ... or other strategies... i mean this
            
            double theta = 2* M_PI * (double)y / ((double) width );
            double sinVal = sinf( theta ) ; //use sin function for double precision, sinf for single precision
            int y1 = ((static_cast<int>(y - x + sc * sinVal) % width) + width) % width  ;  // doing this because % operator in cuda cant do right operation for negative operands
            int x1 = ((static_cast<int>( x - sc * sinVal) % width ) + width ) % width  ;

            //printf("pixel in location (%d,%d), now goes to location (%d,) \n", x,y, x1 ) ;
            output[y1 * width + x1 ] = input[y * width + x] ;
            //output[y1 * width + x1] = 255;
        }
    }
}

__global__ void confusionKernel( const unsigned char *input, unsigned char * output, int width, uint64_t sc){

    int tid_x = threadIdx.x ; //0 to 153
    int tid_y = threadIdx.y ; // 0 to 5
    
    int num_blocks = 128; // there are 128 blocks

    int startRow =   blockIdx.x * (width/num_blocks) ;
    //printf("startRow: %d", startRow) ;

    int x_base = tid_x * 5; // starting column index for this particular thread
    int y = startRow + tid_y; // actual row being processed  

    //if ( y >= startRow + 6 ) return ; 

    for(int i = 0; i< 5; i ++){
        int x = x_base + i ;
        if ( x < width ){  // maybe can remove this if with padding in th future ... or other strategies... i mean this
            int y1 = (y + x) % width ;
            double theta = 2* M_PI * (double)y1 / ((double) width );
            double sinVal = sinf( theta ) ; //use sin function for double precision, sinf for single precision
            double op = x + (((double)sc) * sinVal);
            int x1 = ( ( (static_cast<int>( op ) % width ) + width ) % width ) ;

            printf("pixel in location (%d,%d), now goes to location (%d,) \n", x,y, x1 ) ;
            output[y1 * width + x1 ] = input[y * width + x] ;
            //output[y1 * width + x1] = 255;
        }
    }

}

void confusionOpWrapper( unsigned char *input, unsigned char *output, int width , int height, uint64_t sc, int performInverseConfusion ) {

    const int subframe_height = 6;
    const int total_pixels = width * height ;
    const int num_subframes = height / subframe_height ; //128 considering 768 images

    unsigned char *d_input;
    unsigned char *d_output;

    cudaMalloc(&d_input, total_pixels * sizeof( unsigned char ) ) ;
    cudaMalloc(&d_output, total_pixels * sizeof( unsigned char ) ) ;

    cudaMemcpy(d_input, input, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice) ;

    dim3 blocksPerGrid(128);
    dim3 threadsPerBlock(154,6) ; // maximum number of threads per block that can be allocated.

    if(performInverseConfusion == 0){
        confusionKernel<<< blocksPerGrid, threadsPerBlock >>> ( d_input, d_output, width, sc); 
    } else {
        inverseConfusionKernel <<<  blocksPerGrid, threadsPerBlock >>> (d_input, d_output, width, sc) ;// used to check if indeed we get the starting image back 
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, total_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}