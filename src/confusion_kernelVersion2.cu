#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/confusion_kernel.hpp"
#include <math.h>
#include <iostream>

#define NUM_CHANNELS 3

__global__ void inverseConfusionKernel ( const unsigned char *input, unsigned char *output, int width, int subframe_height, int pixelsPerThread, uint64_t sc ){
    
    int tid_x = threadIdx.x ; //0 to 31
    int tid_y = threadIdx.y ; // 0 to 31
    
    int num_blocks_grey = (  gridDim.x ); // / NUM_CHANNELS ; // there are 128 blocks

    int channelOffset = blockIdx.x / num_blocks_grey ;

    int startRow = ( blockIdx.x % num_blocks_grey ) * (width/num_blocks_grey) ;
    //printf("startRow: %d", startRow) ;

    int y = tid_y ; // actual row being processed  
    int x = tid_x ;

    int SubframeHeight = 32 ;

    int i, j ;

    y = tid_y + 32*0;

    x = tid_x + 32*0;

    float theta = 2* (float)y / ((float) subframe_height );
    float sinVal = sinpif( theta ) ; //use sin function for double precision, sinf for single precision
    int y1 = ((static_cast<int>( (float)y - (float)x + (float)sc * sinVal) % subframe_height) + subframe_height) % subframe_height  ;  // doing this because % operator in cuda cant do right operation for negative operands
    int x1 = ((static_cast<int>( (float)x - (float)sc * sinVal) % subframe_height ) + subframe_height ) % subframe_height  ;

    for(int c = 0; c<NUM_CHANNELS; c++){
        output[ NUM_CHANNELS * ( width*( blockIdx.y*SubframeHeight + y1 ) + ( blockIdx.x * SubframeHeight ) + x1 ) + c ] = 
        
        input[ NUM_CHANNELS * ( width*( blockIdx.y*SubframeHeight + y ) + ( blockIdx.x * SubframeHeight ) + x ) + c ] ;
    }

    __syncthreads();

}

__global__ void confusionKernel( unsigned char *input, unsigned char * output, int width, int subframe_height, int pixelsPerThread, uint64_t sc){

    int x = threadIdx.x ; //0 to 31
    int y = threadIdx.y ; // 0 to 31

    int SubframeHeight = 32 ; 

    int gx = blockIdx.x * subframe_height + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;
    int gIndex = (gy * width + gx) * 3; //width of the entire tile

    //int i, j; 
    //int boundary = (int)ceilf( ((float)subframe_height)/32.0) ;

    extern __shared__ uchar3 tile[] ;

    uchar3 *tile1 = tile ;
    uchar3 *tile2 = tile + subframe_height * subframe_height ;

    uchar3 pix = reinterpret_cast<uchar3*>(input)[ (gy + (32*0) ) * width + (gx + (32*0) ) ] ;
    tile1[ (threadIdx.y + 32*0) * subframe_height + threadIdx.x + 32*0 ] = pix; 


    int y1 = (y + x ) & (subframe_height - 1); //% subframeWidth; //& (subframeWidth - 1) is equivalent to a modulo (%) operation
    float theta = 2.0 * (float)y1 / ( (float) SubframeHeight ) ;
    float sinVal = sinpif(theta) ;
    float op = (float)x + ( ( ( float ) sc) * sinVal ) ;
    int x1 = ( ( (static_cast<int>( op ) & (subframe_height - 1) ) + (subframe_height) ) & (subframe_height - 1) ) ; 


    tile2[ y1 * subframe_height + x1].x = tile1[ y * subframe_height + x].x ;
    tile2[ y1 * subframe_height + x1].y = tile1[ y * subframe_height + x].y ;
    tile2[ y1 * subframe_height + x1].z = tile1[ y * subframe_height + x].z ;

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0 ] = tile2[ y * subframe_height + x ].x ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1 ] = tile2[ y * subframe_height + x ].y ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2 ] = tile2[ y * subframe_height + x ].z ;
            
    __syncthreads();
            
}

void confusionOpWrapper( unsigned char *input, unsigned char * d_input, unsigned char *output, unsigned char *d_output, int width , int height, int subframe_height, uint64_t sc, int performInverseConfusion, cudaStream_t stream1 ) {

    int SUBFRAMEHEIGHT = 32 ;

    int total_pixels = width * height ;
    int dimThreadsXdir = SUBFRAMEHEIGHT ;
    int dimThreadsYdir = SUBFRAMEHEIGHT ;
    
    if (SUBFRAMEHEIGHT >= 32 ) {
        dimThreadsXdir = 32 ;
        dimThreadsYdir = 32 ; //maximum allowed as maximum threads that can be allocated is 32 * 32 = 1024
    }

    int pixelsPerThread = ceilf(( SUBFRAMEHEIGHT * SUBFRAMEHEIGHT ) /1024.0) ;

    const int num_subframes = total_pixels / (SUBFRAMEHEIGHT * SUBFRAMEHEIGHT) ;

   // printf( "number of subframes is %d\n", num_subframes) ;

    int dimBlockXdir = width / SUBFRAMEHEIGHT  ;
    int dimBlockYdir = height / SUBFRAMEHEIGHT ;

    //cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice) ;

    dim3 blocksPerGrid(dimBlockXdir, dimBlockYdir); // * NUM_CHANNELS ) ; // * NUM_CHANNELS
    dim3 threadsPerBlock(dimThreadsXdir, dimThreadsYdir ) ; // maximum number of threads per block that can be allocated.

    //size_t sharedMemTileSize = width * subframe_height * NUM_CHANNELS * sizeof( unsigned char ) ;

    size_t sharedMemPerBlock = 2 * SUBFRAMEHEIGHT * SUBFRAMEHEIGHT * sizeof(uchar3) ;
    
   // printf("\n SubframeHeight : %d \n", SUBFRAMEHEIGHT ) ;

    if(performInverseConfusion == 0) {
        confusionKernel<<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock, stream1 >>> ( d_input, d_output, width, SUBFRAMEHEIGHT, pixelsPerThread, sc);  
    } else {
        inverseConfusionKernel <<<  blocksPerGrid, threadsPerBlock , 0,  stream1 >>> (d_input, d_output, width, SUBFRAMEHEIGHT, pixelsPerThread, sc) ;// used to check if indeed we get the starting image back 
    }
    //cudaDeviceSynchronize() ;

    //cudaMemcpy(output, d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyDeviceToHost);

}