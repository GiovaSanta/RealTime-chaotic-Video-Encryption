#include <iostream>
#include <stdint.h>
#include "../include/diffusion_kernel.hpp"

#define NUM_CHANNELS 3

//#define TOTALPIXELSSUBFRAME 4608 //the specifc subframe at focus is 6 * 768 pixels in total

__device__ void diffusionSeq(int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, const unsigned char *input, const unsigned char *byte_stream, unsigned char *output, int channelOffset, int performInverseDiffusion ){
    
    unsigned char b_byte ;

    if(performInverseDiffusion == 0){
        printf("DIFFUSION RESULTS: \n");
        for ( int y = startRow ; y <= endRow ; y ++ ) {
            for( int x = startCol ; x <= endCol ; x ++ ) {
                for(int c = 0; c < NUM_CHANNELS; c ++ ) {
                    b_byte = byte_stream[ ( y * width + x ) * NUM_CHANNELS + c];
                    if( (y == startRow) && (x == startCol) ) {
                        output[ (y*width+x)*NUM_CHANNELS + c] = b_byte ^ ( input[ (y * width + x) * NUM_CHANNELS + c] + b_byte ) ^ sd ;
                        //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, sd, output[ (y*width + x) * NUM_CHANNELS + c]);
                    } else if ( x == startCol ) {
                        output[ (y*width+x)*NUM_CHANNELS + c] = b_byte ^ ( input[ (y * width + x) * NUM_CHANNELS + c] + b_byte ) ^  output[ ( (y-1)*width + (endCol) )*NUM_CHANNELS + c] ;
                        
                    }else {
                        output[ (y*width+x)*NUM_CHANNELS + c] = b_byte ^ ( input[ (y * width + x) * NUM_CHANNELS + c] + b_byte ) ^ output[ (y*width + (x) )*NUM_CHANNELS + c -1] ;
                        //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
                    }
                    //output[y*width*3 + x*3 ] = 100;    
                }
            }
        }    
    } else { // doing inverse of diffusion .. remember that in the inverse Diffusion operation the input is the ciberSubFrame, the output will be the input subframe which was provided to the diffusion step
        printf("INVERSE DIFFUSION RESULTS: \n");
        for ( int y = endRow ; y >= startRow ; y -- ) {
            for( int x = endCol ; x >= startCol ; x -- ) {
                for( int c = 0; c < NUM_CHANNELS; c ++ )  {
                    b_byte = byte_stream[ (y * width  + x ) * NUM_CHANNELS + c];
                    if( (y == startRow) && (x == startCol) ){
                        output[ (y*width + x) * NUM_CHANNELS + c] = (b_byte ^ input[ (y*width + x) * NUM_CHANNELS + c] ^ sd ) - b_byte ;
                        //printf("(sd involved) output(%d,%d, c=%d) = (%d xor %d xor %d) - %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], sd, b_byte, output[ (y*width + x) * NUM_CHANNELS + c]);
                    } else if ( x == startCol ){
                        output[ (y*width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ ( y*width + x) * NUM_CHANNELS + c] ^  input[ ( (y-1)*width + (endCol ) )*NUM_CHANNELS + c ] ) - b_byte ; 
                    } else{
                        output[ (y*width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ (y*width + x) * NUM_CHANNELS + c] ^  input[ ( y*width + (x ) )*NUM_CHANNELS + c - 1 ] ) - b_byte ;
                        //printf("output(%d,%d, c= %d) = (%d xor %d xor %d) - %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], input[ ( y*width + (x ) )*NUM_CHANNELS + c -1],  b_byte, output[ (y*width + x) * NUM_CHANNELS + c] );

                        //output[y*width+x] =255;
                    }    
                } 
            } 
        }
    }
}

__global__ void diffusionKernel(const unsigned char *input, unsigned char *output, const unsigned char *byte_stream, int width, int height, int subframe_height, int num_subframes, int performInverseDiffusion, unsigned char * d_sd_array ) {

    // will worry about parametrizing better as soon as complete encrption process for the specific case of the 768x768 greyscale frame

    //int tid_x = threadIdx.x ; //0 to 153
    //int tid_y = threadIdx.y ; // 0 to 5

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid = threadIdx.x;
    int channelOffset = blockId_x / gridDim.x ;

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    //if (tid == 0 ){

        unsigned char sd;

        int startRow =  blockId_y  * subframe_height ;  //startRow of the subframe
        int endRow =  startRow + subframe_height - 1 ;

        int startCol = blockId_x * subframe_height;
        int endCol = startCol + subframe_height - 1 ;
        //printf ( "startRow of blockId %d: %d \n", blockId, startRow) ;

        sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;
    
    //if( (blockId_y == 0 ) ){
        if(performInverseDiffusion == 0){
            // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
            
                int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

                printf("\n sd_block X : = %d \n", sd_block_x) ;
                int sd_block_y = (blockId_y + ((blockId_x + 1)/gridDim.x)) % gridDim.y ;

                printf("\n sd block Y : = %d \n ", sd_block_y);
                int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                            ( sd_block_x * subframe_height )    +
                            ( gridDim.x * subframe_height * (subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;
                
                printf("\n sd coordinate: %d\n ", sd_coordinate );

                printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
                sd = input[ sd_coordinate  ] ;

                /*for(int i = 0; i < width * height * NUM_CHANNELS; i ++){
                    printf("\npos %d -> %d, ", i,  input[i]);
                }*/

                d_sd_array[ (blockId_y * gridDim.x) + blockId_x ] = sd ;    
            }
        
        diffusionSeq(startRow, endRow , startCol, endCol, width, sd, input, byte_stream, output , channelOffset, performInverseDiffusion  ); //this is the sequential version based oon chebishev on the paper.. unfutunatly it is nonParallelizable
    
    
    printf("\nsd of block(%d, %d)= %d\n", blockId_x, blockId_y, d_sd_array[(blockId_y * gridDim.x + blockId_x)]) ;
    //}
    //__syncthreads();
    
}

void diffusionOpWrapper( unsigned char *input, unsigned char *output, unsigned char *byteStreamFinal, int width, int height, int subframe_height, int performInverseDiffusion, std::vector<unsigned char>& sd_array) {

    const int total_pixels = width * height ;
    const int num_subframesXdir = height / subframe_height ;
    const int num_subframesYdir = height / subframe_height ;

    printf("\n num_subframesXdir : %d\n", num_subframesXdir);
    printf("\n num_subframesYdir : %d\n", num_subframesYdir);

    unsigned char *d_input ;
    unsigned char *d_output ;
    unsigned char *d_byteStream ; // space which contains all of the byte streams that have to be applied to all of the subframes...

    unsigned char *d_sd_array ;

    //printf("total_pixels: %d\n", total_pixels) ;
    //printf("num_subframes: %d\n", num_subframes) ;

    cudaMalloc(&d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    
    cudaMalloc(&d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    
    cudaMalloc(&d_byteStream, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;

    cudaMalloc( (void **) &d_sd_array, num_subframesXdir * num_subframesYdir * sizeof( unsigned char) ) ;

    cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyHostToDevice) ;
    
    cudaMemcpy(d_byteStream, byteStreamFinal, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyHostToDevice) ;

    if( performInverseDiffusion == 1 ){
        printf("sd values used for inverse diffusion\n");
        for(int i = 0; i < sd_array.size(); i ++){
            printf("% d", sd_array[i]); 
        }
        cudaMemcpy( d_sd_array, sd_array.data(), num_subframesXdir * num_subframesYdir * sizeof( unsigned char ), cudaMemcpyHostToDevice ) ;
    }

    //printf(".......");

    //for now exploiting the parallelism just at the block level ... and each block will run the algorithm serially...
    //will have to modify when optimizing.... either try to find another algorithm for diffusion different from the paper online which is parallelizable....
    //..or stick to this and just try to obtain the correct configurations of blocks which do make obtain the best performance... can see that while profiling i believe

    dim3 blocksPerGrid( num_subframesXdir , num_subframesYdir ) ;
    
    //dim3 threadsPerBlock(154,6) ; // maximum number of threads per block that can be allocated. // same way as confusion
    dim3 threadsPerBlock( 1 ) ;

    diffusionKernel<<< blocksPerGrid, threadsPerBlock >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, num_subframesXdir* num_subframesYdir, performInverseDiffusion, d_sd_array );
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyDeviceToHost);
    
    if( performInverseDiffusion == 0) {
        sd_array.resize( num_subframesXdir * num_subframesYdir ) ;
        cudaMemcpy( sd_array.data(), d_sd_array,  num_subframesXdir * num_subframesYdir * sizeof(unsigned char), cudaMemcpyDeviceToHost ) ;
        for(int i = 0; i < sd_array.size(); i ++){
            printf(" %d ", sd_array[i]);
        }
    }

    cudaFree(d_input);
    cudaFree(d_byteStream);
    cudaFree(d_output);
    cudaFree(d_sd_array);

}