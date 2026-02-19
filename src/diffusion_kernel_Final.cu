#include <iostream>
#include <stdint.h>
#include "../include/diffusion_kernel.hpp"
#include <cuda_profiler_api.h>

#define ROUNDS 5
#define NUM_CHANNELS 3

//#define TOTALPIXELSSUBFRAME 4608 //the specifc subframe at focus is 6 * 768 pixels in total


__global__ void invdiffusionKernelv7( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_width + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    //if (gx >= width || gy >= height)
    //return ;

    extern __shared__ unsigned char  tileinv7[] ;

    //__shared__ unsigned char sd[8] ;

    unsigned char *tile1 = tileinv7 ;
    unsigned char *tile2 = tileinv7 + subframe_height * subframe_width * NUM_CHANNELS ;
    uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ; 
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;
    
    //unsigned char *tile_byteStream = tile2 + subframe_height * subframe_width ;
    //__shared__ unsigned char tile_byteStream[144] ;
    
    unsigned char safePixB =  (input)[ (gy * width + gx) * NUM_CHANNELS + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ (gy * width + gx) * NUM_CHANNELS + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ (gy * width + gx) * NUM_CHANNELS + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ; 

    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    if (  tid_x == 0 || ( !(tid_x & 1) ) ) {
            
            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

            int total = 2 ;

            unsigned char b_byte_B = tile_byteStream[ tid_y * subframe_width + tid_x ].x ; 
    
            unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
            unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
            unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;

            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

            __syncthreads() ;

            for ( int i = tid_y * subframe_width + tid_x + 1 ; i < (tid_y * subframe_width + tid_x + total) ;  i ++ ) {

                int base = i * 3 ;
            
                unsigned char b_byte_B = tile_byteStream[ i ].x ;

                tile2[ base + 0] = (b_byte_B ^  tile1[ base + 0 ] ^  tile1[ base - 1 ] ) - b_byte_B ;
                tile2[ base + 1] = (b_byte_B ^  tile1[ base + 1 ] ^  tile1[ base + 0 ] ) - b_byte_B ;
                tile2[ base + 2] = (b_byte_B ^  tile1[ base + 2 ] ^  tile1[ base + 1 ] ) - b_byte_B ;                     
            } 
        }

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2] ;    

}
__global__ void diffusionKernelv7 ( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array, int round ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;

    int gx = blockIdx.x * subframe_width + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;
    int index = ( gy * width + gx ) * NUM_CHANNELS ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    extern __shared__ unsigned char tilediffv7[] ;

    unsigned char *tile1 = tilediffv7 ;
    unsigned char *tile2 = tile1 + subframe_height * subframe_width * NUM_CHANNELS ;
    //unsigned char *tile_byteStream = tile2 + subframe_height * subframe_width ;
    uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ;
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;

    unsigned char safePixB =  (input)[ index + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ index + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ index + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ;

    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ (gy * width + gx) ] ;
    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    if(  tid_x == 0 || (!(tid_x & 1)) ) {

            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ];

            if( round == 1 ) { //all rounds use the same set of sd values
            
                int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

                int sd_block_y = ( blockId_y + ( (blockId_x + 1) / gridDim.x ) ) % gridDim.y ;

                int sd_coordinate = (( sd_block_y * width * subframe_height ) + 
                                     ( sd_block_x * subframe_width ) + 
                                     ( subframe_height - 1 ) * width + subframe_width - 1) * NUM_CHANNELS ;
                
                sd = input[ sd_coordinate ] ;
                d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ] = sd ; 

            }

        __syncthreads();
    
        unsigned char b_byte_B = tile_byteStream[ tid_y * subframe_width + tid_x ].x ;
    
        unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
        unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
        unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;
    
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^ sd ; 
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = b_byte_B ^ ( in_G_component + b_byte_B ) ^ sd ; 
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = b_byte_B ^ ( in_R_component + b_byte_B ) ^ sd ;

        __syncthreads();

        int  i = tid_y * subframe_width + tid_x + 1;  

        int base = i * 3;
        
        b_byte_B = tile_byteStream[ i ].x ;

        in_B_component = tile1[ base + 0 ] ; 
        in_G_component = tile1[ base + 1 ] ; 
        in_R_component = tile1[ base + 2 ] ; 

        tile2[ base + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^ tile2[ base - 1 ] ;
        tile2[ base + 1 ] = b_byte_B ^ ( in_G_component + b_byte_B ) ^ tile2[ base + 0 ] ;
        tile2[ base + 2 ] = b_byte_B ^ ( in_R_component + b_byte_B ) ^ tile2[ base + 1 ] ;                    

        }

    __syncthreads();

    output[ index + 0 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] ;
    output[ index + 1 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] ;
    output[ index + 2 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] ;

}

void diffusionOpWrapper( unsigned char *d_byteStream, unsigned char * d_input, unsigned char *input, unsigned char * d_output, unsigned char *output, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion,  unsigned char *d_sd_array, cudaStream_t stream1 ) {

    const int total_pixels = width * height ;  // 921600, 960x960
    const int num_subframesXdir = width / subframe_width ; // 80, 960x960
    const int num_subframesYdir = height / subframe_height ; // 120, 960x960

    static int round = 0 ;

    ( performInverseDiffusion == 0 ) ? round ++ : round -- ;


    dim3 blocksPerGrid( num_subframesXdir , num_subframesYdir ) ;
    dim3 threadsPerBlock( subframe_width , subframe_height ) ;

    size_t sharedMemPerBlock = 3 * subframe_height * subframe_width * NUM_CHANNELS * sizeof ( unsigned char ) ;

    if( performInverseDiffusion == 0 ) { // launch diffusion kernel
        //printf("executing diffusion Kernel...\n") ;
        diffusionKernelv7 <<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock, stream1 >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, subframe_width, performInverseDiffusion, d_sd_array, round );
    } else { // launch inverse diffusion kernel 
        invdiffusionKernelv7 <<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock, stream1 >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, subframe_width, performInverseDiffusion, d_sd_array );
    } 
    
}