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

    //printf("boundary: %d\n", (int)ceilf( ((float)subframeHeight)/32.0) );

    //for( i = 0; i < (int)ceilf( ((float)subframe_height)/32.0); i ++ ){

        y = tid_y + 32*0;

        //for( j=0; j < (int)ceilf( ((float)subframe_height)/32.0); j++ ){

            x = tid_x + 32*0;

            float theta = 2* (float)y / ((float) subframe_height );
            float sinVal = sinpif( theta ) ; //use sin function for double precision, sinf for single precision
            int y1 = ((static_cast<int>( (float)y - (float)x + (float)sc * sinVal) % subframe_height) + subframe_height) % subframe_height  ;  // doing this because % operator in cuda cant do right operation for negative operands
            int x1 = ((static_cast<int>( (float)x - (float)sc * sinVal) % subframe_height ) + subframe_height ) % subframe_height  ;

            //printf("pixel in location (%d,%d), now goes to location (%d,) \n", x,y, x1 ) ;
            for(int c = 0; c<NUM_CHANNELS; c++){
                output[ NUM_CHANNELS * ( width*( blockIdx.y*SubframeHeight + y1 ) + ( blockIdx.x * SubframeHeight ) + x1 ) + c ] = 
        
                input[ NUM_CHANNELS * ( width*( blockIdx.y*SubframeHeight + y ) + ( blockIdx.x * SubframeHeight ) + x ) + c ] ;
            }
        //}
    //}
}

__global__ void confusionKernel( unsigned char *input, unsigned char * output, int width, int subframe_height, int pixelsPerThread, uint64_t sc){

    int x = threadIdx.x ; //0 to 31
    int y = threadIdx.y ; // 0 to 31

   int SubframeHeight = 32 ; 

    /*if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("kernel received subframe_height = %d, block size (%d,%d), grid (%d,%d)\n",
           subframe_height, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    }*/
    
    //int num_subframes =(  gridDim.x * gridDim.y ); //900
    
    //int startRow =   ( blockIdx.x % num_subframes ) * ( width/num_subframes) ;
    //printf("startRow: %d", startRow) ;
    
    //int x = tid_x ; // starting column index for this particular thread //3 is the number of channels which must be taken into account

    //int y = tid_y ;   // actual row being processed  
                                // (and 1024 is the maximum of threads that can be allocated per block), then each thread in for a particular block will most likely 
                                // manage more then one pixel

    //if ( x < subframeWidth ){  // maybe can remove this if with padding in th future ... or other strategies... i mean this

    int gx = blockIdx.x * subframe_height + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;
    int gIndex = (gy * width + gx) * 3; //width of the entire tile

    //int i, j; 
    //int boundary = (int)ceilf( ((float)subframe_height)/32.0) ;

    extern __shared__ uchar3 tile[] ;

    uchar3 *tile1 = tile ;
    uchar3 *tile2 = tile + subframe_height * subframe_height ;

    //for( i=0; i< boundary; i++){   //niente.. non funziona l upgrade con la shared memory per quando udo subframes bigger then 32 tipo 64 :(  
        //for( j = 0; j < boundary; j++ ){
            uchar3 pix = reinterpret_cast<uchar3*>(input)[ (gy + (32*0) ) * width + (gx + (32*0) ) ] ;
            tile1[ (threadIdx.y + 32*0) * subframe_height + threadIdx.x + 32*0 ] = pix; 
        //}
    //}

   // for ( i = 0; i< boundary ; i++){
        
        //y = tid_y + 0*32 ;
        //printf("y = %d\n", y) ;
        //for ( j = 0; j< boundary ; j++){
                
            //x = tid_x + 0*32 ;
            //printf("x = %d\n", x) ;
            int y1 = (y + x ) & (subframe_height - 1); //% subframeWidth; //& (subframeWidth - 1) is equivalent to a modulo (%) operation
            float theta = 2.0 * (float)y1 / ( (float) SubframeHeight ) ;
            float sinVal = sinpif(theta) ;
            float op = (float)x + ( ( ( float ) sc) * sinVal ) ;
            int x1 = ( ( (static_cast<int>( op ) & (subframe_height - 1) ) + (subframe_height) ) & (subframe_height - 1) ) ; 

            /*for(int c = 0; c < NUM_CHANNELS; c ++){

                output[ NUM_CHANNELS * ( width*( blockIdx.y*subframeHeight + y1 ) + ( blockIdx.x * subframeWidth ) + x1 ) + c ] = 
         
                input[ NUM_CHANNELS * ( width*( blockIdx.y*subframeHeight + y ) + ( blockIdx.x * subframeWidth ) + x ) + c ] ;

            } */

            tile2[ y1 * subframe_height + x1].x = tile1[ y * subframe_height + x].x ;
            tile2[ y1 * subframe_height + x1].y = tile1[ y * subframe_height + x].y ;
            tile2[ y1 * subframe_height + x1].z = tile1[ y * subframe_height + x].z ;
        //}
        __syncthreads();

        // coalesced writes to global
        //output[ NUM_CHANNELS * ( width * ( blockIdx.y * subframe_height + y ) + ( blockIdx.x * subframe_height ) + x ) + 0 ] = tile2[ y * subframe_height + x ].x ;
        //output[ NUM_CHANNELS * ( width * ( blockIdx.y * subframe_height + y ) + ( blockIdx.x * subframe_height ) + x ) + 1 ] = tile2[ y * subframe_height + x ].y ;
        //output[ NUM_CHANNELS * ( width * ( blockIdx.y * subframe_height + y ) + ( blockIdx.x * subframe_height ) + x ) + 2 ] = tile2[ y * subframe_height + x ].z ;

        output[ (gy * width + gx) * NUM_CHANNELS + 0 ] = tile2[ y * subframe_height + x ].x ;
        output[ (gy * width + gx) * NUM_CHANNELS + 1 ] = tile2[ y * subframe_height + x ].y ;
        output[ (gy * width + gx) * NUM_CHANNELS + 2 ] = tile2[ y * subframe_height + x ].z ;
            //printf("pixel in location (%d,), now goes to location (%d,%d) \n", x, x1,y1 ) ;
            //output[y1 * width + x1 ] = input[y1 * width + x1  ] ; 
            //}
        //__syncthreads();
    //}            
}

void confusionOpWrapper( unsigned char *input, unsigned char *output, int width , int height, int subframe_height, uint64_t sc, int performInverseConfusion ) {

    int SUBFRAMEHEIGHT = 32 ;

    int total_pixels = width * height ;
    int dimThreadsXdir = SUBFRAMEHEIGHT ;
    int dimThreadsYdir = SUBFRAMEHEIGHT ;
    
    if (SUBFRAMEHEIGHT >= 32 ) {
        dimThreadsXdir = 32 ;
        dimThreadsYdir = 32 ; //maximum allowed as maximum threads that can be allocated is 32 * 32 = 1024
    }

    int pixelsPerThread = ceilf(( SUBFRAMEHEIGHT * SUBFRAMEHEIGHT ) /1024.0) ;

   // printf("pixelsPerThread: %d\n", pixelsPerThread ) ;

   // printf("width is %d, height is %d\n", width, height) ;

    unsigned char *d_input ;
    unsigned char *d_output ;

    //printf("dimthreadsXdir is %d, dimthreadsYdir is %d\n", dimThreadsXdir, dimThreadsYdir) ; // each thread corresponds to a pixel

    const int num_subframes = total_pixels / (SUBFRAMEHEIGHT * SUBFRAMEHEIGHT) ;

   // printf( "number of subframes is %d\n", num_subframes) ;

    int dimBlockXdir = width / SUBFRAMEHEIGHT  ;
    int dimBlockYdir = height / SUBFRAMEHEIGHT ;
    
    //printf( "blocks in x direction : %d\n", dimBlockXdir) ;
    //printf( "blocks in y direction : %d\n", dimBlockYdir) ;

    //printf("numThreads allocated in xDir : %d \n", dimThreadsXdir);
    //printf("numThreads allocated in yDir : %d \n", dimThreadsYdir) ;

    //ignoring pixelsPerthread in this version of the confusion ... 
    //int pixelsPerThread = ceilf( ((float)(subframe_height * width))/((float)(dimThreadsXdir*dimThreadsYdir) ) );// since in this current release the subframes will most likely contain more then 1024 pixels 

    //printf("pixelsPer thread : %d\n", pixelsPerThread) ;

    cudaMalloc(&d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc(&d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;

    cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice) ;

    dim3 blocksPerGrid(dimBlockXdir, dimBlockYdir); // * NUM_CHANNELS ) ; // * NUM_CHANNELS
    dim3 threadsPerBlock(dimThreadsXdir, dimThreadsYdir ) ; // maximum number of threads per block that can be allocated.

    //size_t sharedMemTileSize = width * subframe_height * NUM_CHANNELS * sizeof( unsigned char ) ;

    //cudaEvent_t start, stop ;
    //cudaEventCreate(&start) ;
    //cudaEventCreate(&stop ) ;

    size_t sharedMemPerBlock = 2 * SUBFRAMEHEIGHT * SUBFRAMEHEIGHT * sizeof(uchar3) ;
    
   // printf("\n SubframeHeight : %d \n", SUBFRAMEHEIGHT ) ;

    if(performInverseConfusion == 0) {
        confusionKernel<<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock >>> ( d_input, d_output, width, SUBFRAMEHEIGHT, pixelsPerThread, sc);  
        /*cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        }*/
    } else {
        inverseConfusionKernel <<<  blocksPerGrid, threadsPerBlock >>> (d_input, d_output, width, SUBFRAMEHEIGHT, pixelsPerThread, sc) ;// used to check if indeed we get the starting image back 
    }
    cudaDeviceSynchronize() ;

    cudaMemcpy(output, d_output, total_pixels * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

}