#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/confusion_kernel.hpp"
#include <math.h>
#include <iostream>

#define NUM_CHANNELS 3

__global__ void inverseConfusionKernel ( const unsigned char *input, unsigned char *output, int width, int subFrameHeight, int pixelsPerThread, uint64_t sc ){
    int tid_x = threadIdx.x ; //0 to 153
    int tid_y = threadIdx.y ; // 0 to 5
    
    int num_blocks_grey = (  gridDim.x ) / NUM_CHANNELS ; // there are 128 blocks

    int channelOffset = blockIdx.x / num_blocks_grey ;

    int startRow = ( blockIdx.x % num_blocks_grey ) * (width/num_blocks_grey) ;
    //printf("startRow: %d", startRow) ;

    int x_base = tid_x * pixelsPerThread; // starting column index for this particular thread
    int y = startRow + tid_y; // actual row being processed  

    //if ( y >= startRow + 6 ) return ; 

    for(int i = 0; i< pixelsPerThread; i ++){
        int x = x_base + i ;
        if ( x < width ){  // maybe can remove this if with padding in th future ... or other strategies... i mean this
            
            double theta = 2* M_PI * (double)y / ((double) width );
            double sinVal = sinf( theta ) ; //use sin function for double precision, sinf for single precision
            int y1 = ((static_cast<int>(y - x + sc * sinVal) % width) + width) % width  ;  // doing this because % operator in cuda cant do right operation for negative operands
            int x1 = ((static_cast<int>( x - sc * sinVal) % width ) + width ) % width  ;

            //printf("pixel in location (%d,%d), now goes to location (%d,) \n", x,y, x1 ) ;
            output[y1 * width*NUM_CHANNELS + x1*NUM_CHANNELS + channelOffset  ] = input[y * width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset ] ;
            //output[y1 * width + x1] = 255;
        }
    }
}

__global__ void confusionKernel( const unsigned char *input, unsigned char * output, int width, int subFrameHeight, int pixelsPerThread, uint64_t sc){

    int tid_x = threadIdx.x ; //0 to 153
    int tid_y = threadIdx.y ; // 0 to 5
    
    int num_blocks_grey =(  gridDim.x ) / NUM_CHANNELS ; // there are 128 blocks for the greyscale version

    int channelOffset = blockIdx.x / num_blocks_grey ;  
    
    int startRow =   ( blockIdx.x % num_blocks_grey ) * ( width/num_blocks_grey) ;
    //printf("startRow: %d", startRow) ;

    int x_base = tid_x * pixelsPerThread ; // starting column index for this particular thread //3 is the number of channels which must be taken into account
    int y = startRow + tid_y; // actual row being processed  

                          // (and 1024 is the maximum of threads that can be allocated per block), then each thread in for a particular block will most likely 
                          // manage more then one pixel

    //if ( y >= startRow + 6 ) return ; 

    //output[y * width*3 + x_base*3 + channelOffset + 2 ] = input[y * width*3 + x_base*3 + channelOffset + 2] ; //used for DEBUG.... if you launch just 128 blocks, you will see just the blue component of the base pixel of each thread (each thread manages 5 pixels stemmming from a base pixel )
                                                                                                    // if were to run it with 128 * 2 blocks, you will see the blue and green components of each base pixels of all the subframes...

    for(int i = 0; i< pixelsPerThread ; i++ ){   // since in this release0 of the encryption flow each thread handles more then one pixel,this for loop right here is needed.
        int x = x_base + i  ;  // assuming channelOffset to be 0 rignt now
        if ( x < width ){  // maybe can remove this if with padding in th future ... or other strategies... i mean this
            int y1 = (y + x ) % (width) ;
            double theta = 2* M_PI * (double)y1 / ((double) width );
            double sinVal = sinf( theta ) ; //use sin function for double precision, sinf for single precision
            double op = x + (((double)sc) * sinVal);
            int x1 = ( ( (static_cast<int>( op ) % (width) ) + (width) ) % (width) ) ; // sequence of % is done because of cuda inability of "%" for negative operands

            //printf("pixel in location (%d,%d), now goes to location (%d,%d) \n", x,y, x1,y1 ) ;
            //output[y1 * width + x1 ] = input[y1 * width + x1  ] ; 
            
            output[ NUM_CHANNELS*y1 * width + NUM_CHANNELS* x1 + channelOffset ] = input[y * width*NUM_CHANNELS+ x* NUM_CHANNELS + channelOffset ] ;
        }
    }
}

void confusionOpWrapper( unsigned char *input, unsigned char *output, int width , int height, int subframe_height, uint64_t sc, int performInverseConfusion ) {

    const int total_pixels = width * height ;
    const int num_subframes = height / subframe_height ; //128 considering 768 images

    unsigned char *d_input;
    unsigned char *d_output;

    
    int dimThreadsYdir = subframe_height ;
    int dimThreadsXdir =  1024/dimThreadsYdir ; // 1024 is the total number of threads that can be assigned per block on the jetson nano. dimThreadsXdir takes truncated value of such calculation

    printf("numThreads allocated in xDir : %d \n", dimThreadsXdir);
    printf("numThreads allocated in yDir : %d \n", dimThreadsYdir) ;

    int pixelsPerThread = ceilf( ((float)(subframe_height * width))/((float)(dimThreadsXdir*dimThreadsYdir) ) );// since in this current release the subframes will most likely contain more then 1024 pixels 

    printf("pixelsPer thread : %d\n", pixelsPerThread) ;
    cudaMalloc(&d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;

    cudaMalloc(&d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;

    cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice) ;

    dim3 blocksPerGrid(num_subframes * NUM_CHANNELS ); 
    dim3 threadsPerBlock(dimThreadsXdir,dimThreadsYdir) ; // maximum number of threads per block that can be allocated.

    if(performInverseConfusion == 0){
        confusionKernel<<< blocksPerGrid, threadsPerBlock >>> ( d_input, d_output, width, subframe_height, pixelsPerThread, sc); 
    } else {
        inverseConfusionKernel <<<  blocksPerGrid, threadsPerBlock >>> (d_input, d_output, width, subframe_height, pixelsPerThread, sc) ;// used to check if indeed we get the starting image back 
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, total_pixels * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}