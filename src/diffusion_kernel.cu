#include <iostream>
#include <stdint.h>
#include "../include/diffusion_kernel.hpp"

#define NUM_CHANNELS 3

//#define TOTALPIXELSSUBFRAME 4608 //the specifc subframe at focus is 6 * 768 pixels in total

__device__ void diffusionSeq(int startRow, int endRow, int width, unsigned char sd, const unsigned char *input, const unsigned char *byte_stream, unsigned char *output, int channelOffset, int performInverseDiffusion ){
    
    unsigned char b_byte ;

    if(performInverseDiffusion == 0){
        for ( int y = startRow ; y <= endRow ; y++ ) {
            for( int x = 0; x < width ; x ++){
                b_byte = byte_stream[ y * width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset];
                if( y == startRow && x == 0) {
                    output[y*width*NUM_CHANNELS+x*NUM_CHANNELS + channelOffset] = b_byte ^ ( input[y*width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset] + b_byte ) ^ sd ;
                } else {
                    output[y*width*NUM_CHANNELS+x*NUM_CHANNELS + channelOffset ] = b_byte ^ ( input[y*width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset] + b_byte ) ^ output[ y*width*NUM_CHANNELS +(x-1)*NUM_CHANNELS + channelOffset] ;
                }
                //output[y*width*3 + x*3 ] = 100;
            }
        }    
    } else{ // doing inverse of diffusion .. remember that in the inverse Diffusion operation the input is the ciberSubFrame, the output will be the input subframe which was provided to the diffusion step
        for ( int y = endRow ; y >= startRow ; y-- ) {
            for( int x = width - 1; x >=0 ; x -- ){
                b_byte = byte_stream[ y * width * NUM_CHANNELS + x * NUM_CHANNELS + channelOffset];
                if( y == startRow && x == 0 ){
                    output[y*width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset] = (b_byte ^ input[y*width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset] ^ sd ) - b_byte ;
                }else {
                    output[y*width*NUM_CHANNELS +x*NUM_CHANNELS + channelOffset] = (b_byte ^  input[y*width*NUM_CHANNELS + x*NUM_CHANNELS + channelOffset] ^  input[ y*width*NUM_CHANNELS + (x - 1)*NUM_CHANNELS + channelOffset ] ) - b_byte ;
                    //output[y*width+x] =255;
                } 
            }
        }
    }
}

__global__ void diffusionKernel(const unsigned char *input, unsigned char *output, const unsigned char *byte_stream, int width, int height, int num_subframes, int performInverseDiffusion) {

    // will worry about parametrizing better as soon as complete encrption process for the specific case of the 768x768 greyscale frame

    //int tid_x = threadIdx.x ; //0 to 153
    //int tid_y = threadIdx.y ; // 0 to 5

    int blockId = blockIdx.x ;
    int tid = threadIdx.x;
    int channelOffset = blockId/ num_subframes ;

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    if (tid == 0 ){

        int startRow =   (blockId % num_subframes) * (width/num_subframes) ;  //startRow of the subframe
        int endRow = startRow + (width/num_subframes) - 1 ;
        //printf ( "startRow of blockId %d: %d \n", blockId, startRow) ;

        // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
        int sd_row_coordinate_part1 = ( ( ( blockId + 1) % num_subframes )*(width/num_subframes) ) + width/num_subframes - 1  ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 
        unsigned char sd = input[ (sd_row_coordinate_part1*width*NUM_CHANNELS) + (width - 1)*NUM_CHANNELS + channelOffset  ]; 
        
        diffusionSeq(startRow, endRow , width, sd, input, byte_stream, output , channelOffset, performInverseDiffusion  ); //this is the sequential version based oon chebishev on the paper.. unfutunatly it is nonParallelizable
    }

    __syncthreads();
    
    //printf("startRow: %d", startRow) ;

    //int x_base = tid_x * 5; // starting column index for this particular thread
    //int y = startRow + tid_y; // actual row being processed  

    //if ( y >= startRow + 6 ) return ; 


    //i was implementing below following section assuming erroneusly that the diffusion algorithm in the paper was parallelizable 
    /*for(int i = 0; i< 5; i ++){
        int x = x_base + i ;
        if ( x < width ){  // maybe can remove this if with padding in th future ... or other strategies... i mean this
            
            //fetch of a single byte from the byteStream of the specific block at hand...
            byte_b = byte_stream[ y*width  + x ]

            if ( y == startRow AND x == 0 ){
                output[ y * width + x] = byte_b ^ input[ y * width + x + byte_b] ^ sd ;
            } else {
                output[ y * width +  x] = byte_b ^ ( input[ y * width + x] + byte_b ) ^ input[ y* width + x - 1] ; // involve the previous pixel in the diffusion operation
            }

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
    */
}

void diffusionOpWrapper( unsigned char *input, unsigned char *output, unsigned char *byteStreamFinal, int width, int height, int subframe_height, int performInverseDiffusion) {

    const int total_pixels = width * height ;
    const int num_subframes = height / subframe_height ;

    unsigned char *d_input ;
    unsigned char *d_output ;
    unsigned char *d_byteStream ;  // space which contains all of the byte streams that have to be applied to all of the subframes...

    //printf("total_pixels: %d\n", total_pixels) ;
    //printf("num_subframes: %d\n", num_subframes) ;

    cudaMalloc(&d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc(&d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc(&d_byteStream, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;

    cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyHostToDevice) ;
    
    cudaMemcpy(d_byteStream, byteStreamFinal, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyHostToDevice);

    //printf(".......");

    //for now exploiting the parallelism just at the block level ... and each block will run the algorithm serially...
    //will have to modify when optimizing.... either try to find another algorithm for diffusion different from the paper online which is parallelizable....
    //..or stick to this and just try to obtain the correct configurations of blocks which do make obtain the best performance... can see that while profiling i believe

    dim3 blocksPerGrid(num_subframes*NUM_CHANNELS);
    //dim3 threadsPerBlock(154,6) ; // maximum number of threads per block that can be allocated. // same way as confusion
    dim3 threadsPerBlock(1);

    diffusionKernel<<< blocksPerGrid, threadsPerBlock >>> ( d_input, d_output, d_byteStream, width, height, num_subframes, performInverseDiffusion );
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_byteStream);
    cudaFree(d_output);
}