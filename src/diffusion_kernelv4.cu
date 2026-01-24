#include <iostream>
#include <stdint.h>
#include "../include/diffusion_kernel.hpp"
#include <cuda_profiler_api.h>

#define ROUNDS 5
#define NUM_CHANNELS 3

//#define TOTALPIXELSSUBFRAME 4608 //the specifc subframe at focus is 6 * 768 pixels in total

__device__ void diffusionSeq(int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, unsigned char *input, unsigned char *byte_stream, unsigned char *output, int channelOffset, int performInverseDiffusion ){
    
    unsigned char b_byte ;

    //printf("DIFFUSION RESULTS: \n");
    for ( int y = startRow , i = 0; y <= endRow ; y ++ , i++) {
        for( int x = startCol, j = 0 ; x <= endCol ; x ++, j ++ ) {
            for(int c = 0; c < NUM_CHANNELS; c ++ ) {
                b_byte = byte_stream[ ( i * 16 + j ) * NUM_CHANNELS + c];
                if( (y == startRow) && (x == startCol) ) {
                    output[ (i* 16 + j ) * NUM_CHANNELS + c] = b_byte ^ ( input[ (i * 16 + j) * NUM_CHANNELS + c] + b_byte ) ^ sd ;
                    //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, sd, output[ (y*width + x) * NUM_CHANNELS + c]);
                } else if ( x == startCol ) {
                    output[ (i * 16 + j ) * NUM_CHANNELS + c] = b_byte ^ ( input[ (i * 16 + j) * NUM_CHANNELS + c] + b_byte ) ^  output[ ( (i - 1 ) * 16 + (15) ) * NUM_CHANNELS + c ] ;    
                } else {
                    output[ (i * 16 + j) * NUM_CHANNELS + c] = b_byte ^ ( input[ (i * 16 + j) * NUM_CHANNELS + c] + b_byte ) ^ output[ (i * 16 + (j) ) * NUM_CHANNELS + c - 1 ] ;
                    //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
                }
                //output[y*width*3 + x*3 ] = 100;    
            }
        }
    }     
}

__device__ void diffusionSeqv2(int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, uchar3 *input, uchar3 *byte_stream, uchar3 *output, int channelOffset, int performInverseDiffusion, int subframe_height ){
    
    //printf("DIFFUSION RESULTS: \n");
    for ( int y = startRow , i = 0; y <= endRow ; y ++ , i++) {
        for( int x = startCol, j = 0 ; x <= endCol ; x ++, j ++ ) {
                unsigned char b_byte_B = byte_stream[ i * subframe_height + j ].x ;
                unsigned char b_byte_G = byte_stream[ i * subframe_height + j ].y ;
                unsigned char b_byte_R = byte_stream[ i * subframe_height + j ].z ;

                unsigned char in_B_component = input[ i * subframe_height + j ].x ;
                unsigned char in_G_component = input[ i * subframe_height + j ].y ;
                unsigned char in_R_component = input[ i * subframe_height + j ].z ;

            //for( int c = 0; c< NUM_CHANNELS; c ++ ){
                if( (y == startRow) && (x == startCol) ) {
                    output[ i* subframe_height + j ].x = b_byte_B ^ ( in_B_component + b_byte_B ) ^ sd ;
                    output[ i* subframe_height + j ].y = b_byte_G ^ ( in_G_component + b_byte_G ) ^ sd ;
                    output[ i* subframe_height + j ].z = b_byte_R ^ ( in_R_component + b_byte_R ) ^ sd ;
                    //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, sd, output[ (y*width + x) * NUM_CHANNELS + c]);
                }else if ( x == startCol ) {
                    output[ i * subframe_height + j ].x = b_byte_B ^ ( in_B_component + b_byte_B ) ^  output[ (i - 1 ) * subframe_height + (subframe_height-1) ].x ;    
                    output[ i * subframe_height + j ].y = b_byte_G ^ ( in_G_component + b_byte_G ) ^  output[ (i - 1 ) * subframe_height + (subframe_height-1) ].y ;    
                    output[ i * subframe_height + j ].z = b_byte_R ^ ( in_R_component + b_byte_R ) ^  output[ (i - 1 ) * subframe_height + (subframe_height-1) ].z ;    
                
                }else {
                    output[ i * subframe_height + j ].x = b_byte_B ^ ( in_B_component + b_byte_B ) ^ output[ i * subframe_height + j - 1 ].x ;
                    output[ i * subframe_height + j ].y = b_byte_G ^ ( in_G_component + b_byte_G ) ^ output[ i * subframe_height + j - 1 ].y ;
                    output[ i * subframe_height + j ].z = b_byte_R ^ ( in_R_component + b_byte_R ) ^ output[ i * subframe_height + j - 1 ].z ;                    
                    
                    //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
                }
            //}
                //output[y*width*3 + x*3 ] = 100;    
        }
    }     
}

__device__ void diffusionSeqv3( int width, unsigned char sd, unsigned char *input, uchar3 *byte_stream, unsigned char *output, int channelOffset, int performInverseDiffusion, int subframe_height ){

    //printf("DIFFUSION RESULTS: \n");
    
    int total = subframe_height * subframe_height ;
    
    //#pragma unroll
    for ( int  i = 0; i < total ; i++) {

        int base = i * NUM_CHANNELS;
        
        unsigned char b_byte_B = byte_stream[ i ].x ;
        unsigned char b_byte_G = byte_stream[ i ].y ;
        unsigned char b_byte_R = byte_stream[ i ].z ;

        //unsigned char b_byte_B = ((unsigned char*)&byte_stream[i])[0] ;
        //unsigned char b_byte_G = ((unsigned char*)&byte_stream[i])[1] ;
        //unsigned char b_byte_R = ((unsigned char*)&byte_stream[i])[2] ;

        unsigned char in_B_component = input[ base + 0] ;
        unsigned char in_G_component = input[ base + 1] ;
        unsigned char in_R_component = input[ base + 2] ;

        //for( int c = 0; c< NUM_CHANNELS; c ++ ){
        if( i == 0) {
            output[ base + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^ sd ;
            output[ base + 1 ] = b_byte_G ^ ( in_G_component + b_byte_G ) ^ sd ;
            output[ base + 2 ] = b_byte_R ^ ( in_R_component + b_byte_R ) ^ sd ;
            //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, sd, output[ (y*width + x) * NUM_CHANNELS + c]);
        //}else if ( j == 0 ) {
        //    output[ idx * NUM_CHANNELS + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^  output[ ((i - 1 ) * subframe_height + (subframe_height-1)) * NUM_CHANNELS + 0 ] ;    
        //    output[ idx * NUM_CHANNELS + 1 ] = b_byte_G ^ ( in_G_component + b_byte_G ) ^  output[ ((i - 1 ) * subframe_height + (subframe_height-1)) * NUM_CHANNELS + 1 ] ;    
        //    output[ idx * NUM_CHANNELS + 2 ] = b_byte_R ^ ( in_R_component + b_byte_R ) ^  output[ ((i - 1 ) * subframe_height + (subframe_height-1)) * NUM_CHANNELS + 2 ] ;    
                
        } else {
            output[ base + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^ output[ base + 0 - 1 ] ;
            output[ base + 1 ] = b_byte_G ^ ( in_G_component + b_byte_G ) ^ output[ base + 1 - 1 ] ;
            output[ base + 2 ] = b_byte_R ^ ( in_R_component + b_byte_R ) ^ output[ base + 2 - 1 ] ;                    
                    
            //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
        }
        //}
                //output[y*width*3 + x*3 ] = 100; 
    }     
}

__device__ void diffusionSeqv4( int width, unsigned char sd, uchar3 *input, uchar3 *byte_stream, uchar3 *output, int channelOffset, int performInverseDiffusion, int subframe_height ) {

    int total = subframe_height * subframe_height ;

    //#pragma unroll
    for ( int  i = 0; i < total ; i++) {

        //int base = i * NUM_CHANNELS;
        
        unsigned char b_byte_B = byte_stream[ i ].x ;
        //unsigned char b_byte_G = byte_stream[ i ].y ;
        //unsigned char b_byte_R = byte_stream[ i ].z ;

        unsigned char in_B_component = input[ i ].x ;
        unsigned char in_G_component = input[ i ].y ;
        unsigned char in_R_component = input[ i ].z ;

        //for( int c = 0; c< NUM_CHANNELS; c ++ ){
        if( i == 0) {
            output[ i ].x = b_byte_B ^ ( in_B_component + b_byte_B ) ^ sd ;
            output[ i ].y = b_byte_B ^ ( in_G_component + b_byte_B ) ^ sd ;
            output[ i ].z = b_byte_B ^ ( in_R_component + b_byte_B ) ^ sd ;


            //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, sd, output[ (y*width + x) * NUM_CHANNELS + c]);
        //}else if ( j == 0 ) {
        //    output[ idx * NUM_CHANNELS + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^  output[ ((i - 1 ) * subframe_height + (subframe_height-1)) * NUM_CHANNELS + 0 ] ;    
        //    output[ idx * NUM_CHANNELS + 1 ] = b_byte_G ^ ( in_G_component + b_byte_G ) ^  output[ ((i - 1 ) * subframe_height + (subframe_height-1)) * NUM_CHANNELS + 1 ] ;    
        //    output[ idx * NUM_CHANNELS + 2 ] = b_byte_R ^ ( in_R_component + b_byte_R ) ^  output[ ((i - 1 ) * subframe_height + (subframe_height-1)) * NUM_CHANNELS + 2 ] ;    
                
        } else {
            output[ i ].x = b_byte_B ^ ( in_B_component + b_byte_B ) ^ output[ i - 1 ].x ;
            output[ i ].y = b_byte_B ^ ( in_G_component + b_byte_B ) ^ output[ i - 1 ].y ;
            output[ i ].z = b_byte_B ^ ( in_R_component + b_byte_B ) ^ output[ i - 1 ].z ;                    
                    
            //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
        }
        //}
                //output[y*width*3 + x*3 ] = 100; 
    }
}

__device__ void invdiffusionSeq( int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, const unsigned char *input, const unsigned char *byte_stream, unsigned char *output, int channelOffset, int performInverseDiffusion ) {

    unsigned char b_byte ;

    for ( int y = endRow ; y >= startRow ; y -- ) {
        for( int x = endCol ; x >= startCol ; x -- ) {
            for( int c = 0; c < NUM_CHANNELS; c ++ )  {
                b_byte = byte_stream[ (y * width  + x ) * NUM_CHANNELS + c];
                if( (y == startRow) && (x == startCol) ){
                    output[ (y * width + x) * NUM_CHANNELS + c] = (b_byte ^ input[ (y * width + x) * NUM_CHANNELS + c] ^ sd ) - b_byte ;
                    //printf("(sd involved) output(%d,%d, c=%d) = (%d xor %d xor %d) - %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], sd, b_byte, output[ (y*width + x) * NUM_CHANNELS + c]);
                //} else if ( x == startCol ){
                 //   output[ (y * width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ ( y * width + x) * NUM_CHANNELS + c] ^  input[ ( ( y - 1 ) * width + ( endCol ) ) * NUM_CHANNELS + c ] ) - b_byte ; 
                } else{
                    output[ (y * width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ ( y * width + x) * NUM_CHANNELS + c] ^  input[ ( y*width + (x ) )*NUM_CHANNELS + c - 1 ] ) - b_byte ;
                    //printf("output(%d,%d, c= %d) = (%d xor %d xor %d) - %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], input[ ( y*width + (x ) )*NUM_CHANNELS + c -1],  b_byte, output[ (y*width + x) * NUM_CHANNELS + c] );
                    //output[y*width+x] =255;
                }    
            } 
        } 
    }
}

__device__ void invdiffusionSeqv2(  int width, unsigned char sd, unsigned char *input, unsigned char *byte_stream, unsigned char *output, int channelOffset, int performInverseDiffusion, int subframe_height ){
    
    int total = subframe_height * subframe_height ;
    
    //#pragma unroll

    unsigned char b_byte_B = byte_stream[ 0 ] ;
    //unsigned char b_byte_G = tile_byteStream[ 0 ].y ;
    //unsigned char b_byte_R = tile_byteStream[ 0 ].z ;
    
    unsigned char in_B_component = input[ 0 ] ;
    unsigned char in_G_component = input[ 1 ] ;
    unsigned char in_R_component = input[ 2 ] ;
    
    output[ 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ;
    output[ 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ;
    output[ 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

    for ( int i = 1 ; i < total ;  i ++ ) {
        //for( int j = 0 ; j < subframe_height ;  j ++ ) {

            int base = i * 3;
            
            unsigned char b_byte_B = byte_stream[ i ] ;
            //unsigned char b_byte_G = byte_stream[ i * subframe_height + j ].y ;
            //unsigned char b_byte_R = byte_stream[ i * subframe_height + j ].z ;


            //unsigned char b_byte_B = ((unsigned char*)&byte_stream[i])[channelOffset] ;

            //if(  i == 0  ) {
               // output[ base + 0 ] = ( b_byte_B ^ input[ base + 0] ^ sd ) - b_byte_B ;
               // output[ base + 1 ] = ( b_byte_B ^ input[ base + 1] ^ sd ) - b_byte_B ;
               // output[ base + 2 ] = ( b_byte_B ^ input[ base + 2] ^ sd ) - b_byte_B ;
                //printf("(sd involved) output(%d,%d, c=%d) = (%d xor %d xor %d) - %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], sd, b_byte, output[ (y*width + x) * NUM_CHANNELS + c]);
            //} else if ( x == startCol ){
            //   output[ (y * width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ ( y * width + x) * NUM_CHANNELS + c] ^  input[ ( ( y - 1 ) * width + ( endCol ) ) * NUM_CHANNELS + c ] ) - b_byte ; 
            //} else {
                output[ base + 0] = (b_byte_B ^  input[ base + 0 ] ^  input[ base - 1 ] ) - b_byte_B ;
                output[ base + 1] = (b_byte_B ^  input[ base + 1 ] ^  input[ base + 0 ] ) - b_byte_B ;
                output[ base + 2] = (b_byte_B ^  input[ base + 2 ] ^  input[ base + 1 ] ) - b_byte_B ;
                                
                //printf("output(%d,%d, c= %d) = (%d xor %d xor %d) - %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], input[ ( y*width + (x ) )*NUM_CHANNELS + c -1],  b_byte, output[ (y*width + x) * NUM_CHANNELS + c] );
                //output[y*width+x] =255;
            //}     
        //} 
    }
}

__device__ void invdiffusionSeqv3( int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, uchar3 *input, uchar3 *byte_stream, uchar3 *output, int channelOffset, int performInverseDiffusion, int subframe_height ){
    
    for ( int i = 0 ; i < subframe_height ;  i ++ ) {
        for( int j = 0 ; j < subframe_height ;  j ++ ) {
            
            unsigned char b_byte_B = byte_stream[ i * subframe_height + j ].x ;
            unsigned char b_byte_G = byte_stream[ i * subframe_height + j ].y ;
            unsigned char b_byte_R = byte_stream[ i * subframe_height + j ].z ;

            if( ( i == 0 ) && ( j == 0) ) {
                output[ (i * subframe_height + j)  ].x = ( b_byte_B ^ input[ (i * subframe_height + j) ].x ^ sd ) - b_byte_B ;
                output[ (i * subframe_height + j)  ].y = ( b_byte_G ^ input[ (i * subframe_height + j) ].y ^ sd ) - b_byte_G ;
                output[ (i * subframe_height + j)  ].z = ( b_byte_R ^ input[ (i * subframe_height + j) ].z ^ sd ) - b_byte_R ;
                //printf("(sd involved) output(%d,%d, c=%d) = (%d xor %d xor %d) - %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], sd, b_byte, output[ (y*width + x) * NUM_CHANNELS + c]);
            //} else if ( x == startCol ){
            //   output[ (y * width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ ( y * width + x) * NUM_CHANNELS + c] ^  input[ ( ( y - 1 ) * width + ( endCol ) ) * NUM_CHANNELS + c ] ) - b_byte ; 
            } else {
                output[ (i * subframe_height + j) ].x = (b_byte_B ^  input[ ( i * subframe_height + j) ].x ^  input[ ( i* subframe_height + j )  - 1 ].x ) - b_byte_B ;
                output[ (i * subframe_height + j) ].y = (b_byte_G ^  input[ ( i * subframe_height + j) ].y ^  input[ ( i* subframe_height + j )  - 1 ].y ) - b_byte_G ;
                output[ (i * subframe_height + j) ].z = (b_byte_R ^  input[ ( i * subframe_height + j) ].z ^  input[ ( i* subframe_height + j )  - 1 ].z ) - b_byte_R ;
                                
                //printf("output(%d,%d, c= %d) = (%d xor %d xor %d) - %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], input[ ( y*width + (x ) )*NUM_CHANNELS + c -1],  b_byte, output[ (y*width + x) * NUM_CHANNELS + c] );
                //output[y*width+x] =255;
            }     
        } 
    }
}

__global__ void invdiffusionKernel(const unsigned char *input, unsigned char *output, const unsigned char *byte_stream, int width, int height, int subframe_height, int performInverseDiffusion, unsigned char * d_sd_array ) { // inverse diffusion with the shared memory 
    
    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    int channelOffset = blockId_x / gridDim.x ;

    unsigned char sd;

    int startRow = blockId_y  * subframe_height ;  //startRow of the subframe
    int endRow = startRow + subframe_height - 1 ;
    int startCol = blockId_x * subframe_height ;
    int endCol = startCol + subframe_height - 1 ;

    sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

    if( ( tid_x == 0 ) && ( tid_y == 0 ) ) {
        invdiffusionSeq(startRow, endRow , startCol, endCol, width, sd, input, byte_stream, output , channelOffset, performInverseDiffusion  ) ; 
    }

} 

__global__ void diffusionKernelv3(unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int performInverseDiffusion, unsigned char * d_sd_array, int round ) {
    
    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_height + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    //if (gx >= width || gy >= height)
    //return ;

    extern  __shared__  uchar3  tilediffv3[] ;

    uchar3 *tile1 = tilediffv3 ;
    uchar3 *tile2 = tilediffv3 + subframe_height * subframe_height ;
    uchar3 *tile_byteStream = tile2 + subframe_height * subframe_height ;

    tile1[ threadIdx.y * subframe_height + threadIdx.x ] = reinterpret_cast<uchar3*>(input)[ (gy * width + gx) ] ;
     
    //tile1[ threadIdx.y * subframe_height + threadIdx.x ] = reinterpret_cast<uchar3*>(input)[ (globalIndex) ] ; ; 

    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ gy * width + gx ] ;
    tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safe3bytes ;

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    //if (tid == 0 ){

    if( ( tid_x == 0 ) && ( tid_y == 0 ) ) {

    unsigned char sd ;

        if( round == 1 ){ //all rounds use the same set of sd values

            // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
            
            int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

            //printf("\n sd_block X : = %d \n", sd_block_x) ;
            int sd_block_y = (blockId_y + ((blockId_x + 1)/gridDim.x)) % gridDim.y ;

            //printf("\n sd block Y : = %d \n ", sd_block_y);
            int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                        ( sd_block_x * subframe_height )    +
                        ( gridDim.x * subframe_height * (subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;
                
            //printf("\n sd coordinate: %d\n ", sd_coordinate );

            //printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
            sd = input[ sd_coordinate  ] ;
            d_sd_array[ (blockId_y * gridDim.x) + blockId_x ] = sd ; 
        }
        
    //diffusionSeqv4( width, sd, tile1, tile_byteStream, tile2 , channelOffset , performInverseDiffusion, subframe_height ); //this is the sequential version based oon chebishev on the paper.. unfutunatly it is nonParallelizable
    
    int total = subframe_height * subframe_height ;
    
    unsigned char b_byte_B = tile_byteStream[ 0 ].x ;

    uchar3 registerVal = tile1[0];

    unsigned char in_B_component = registerVal.x ;
    unsigned char in_G_component = registerVal.y ;
    unsigned char in_R_component = registerVal.z ;
    
    tile2[ 0 ].x = b_byte_B ^ ( in_B_component + b_byte_B ) ^ sd ;
    tile2[ 0 ].y = b_byte_B ^ ( in_G_component + b_byte_B ) ^ sd ;
    tile2[ 0 ].z = b_byte_B ^ ( in_R_component + b_byte_B ) ^ sd ;

    //#pragma unroll
    for ( int  i = 1; i < total ; i++) {
        
        unsigned char b_byte_B = tile_byteStream[ i ].x ;
        //unsigned char b_byte_G = byte_stream[ i ].y ;
        //unsigned char b_byte_R = byte_stream[ i ].z ;

        uchar3 registerLast = tile2[i-1] ;
        uchar3 registerVal = tile1[i] ;

        /*unsigned char in_B_component =  ;
        unsigned char in_G_component =  ;
        unsigned char in_R_component =  ; */
        
        tile2[ i ].x = b_byte_B ^ ( registerVal.x + b_byte_B ) ^ registerLast.x ;
        tile2[ i ].y = b_byte_B ^ ( registerVal.y + b_byte_B ) ^ registerLast.y ;
        tile2[ i ].z = b_byte_B ^ ( registerVal.z + b_byte_B ) ^ registerLast.z ;                    
                    
    }

    }

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_height + threadIdx.x) ].x ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_height + threadIdx.x) ].y ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_height + threadIdx.x) ].z ;

    //reinterpret_cast<uchar3*>(output)[gy * width + gx] = tile2[threadIdx.y * subframe_height + threadIdx.x] ;

    //printf("\nsd of block(%d, %d)= %d\n", blockId_x, blockId_y, d_sd_array[(blockId_y * gridDim.x + blockId_x)]) ;
    //}
    //__syncthreads();
    
}

__global__ void invdiffusionKernelv3(unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int performInverseDiffusion, unsigned char * d_sd_array ){

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_height + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    if (gx >= width || gy >= height)
    return ;

    extern __shared__ uchar3  tileinv3[] ;

    uchar3 *tile1 = tileinv3 ; // for the output
    uchar3 *tile2 = tileinv3 + subframe_height * subframe_height ; // for the  input
    uchar3 *tile_byteStream = tile2 + subframe_height * subframe_height  ;

    uchar3 safePix =  reinterpret_cast<uchar3*>(input)[ (gy * width + gx) ] ;
    tile1[ (threadIdx.y * subframe_height + threadIdx.x) ] = safePix ; 

    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ gy * width + gx ] ;
    tile_byteStream[  threadIdx.y * subframe_height + threadIdx.x ] = safe3bytes ;

    unsigned char sd ;

    int startRow = blockId_y  * subframe_height ;  //startRow of the subframe
    int endRow = startRow + subframe_height - 1 ;
    int startCol = blockId_x * subframe_height ;
    int endCol = startCol + subframe_height - 1 ;

    sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

    if( ( tid_x == 0 ) && ( tid_y == 0 ) ) {
        //invdiffusionSeqv3(startRow, endRow , startCol, endCol, width, sd, tile1, tile_byteStream, tile2 , channelOffset, performInverseDiffusion, subframe_height  ) ; 
    
    for ( int i = 0 ; i < subframe_height ;  i ++ ) {
        for( int j = 0 ; j < subframe_height ;  j ++ ) {
            
            unsigned char b_byte_B = tile_byteStream[ i * subframe_height + j ].x ;
            //unsigned char b_byte_G = byte_stream[ i * subframe_height + j ].y ;
            //unsigned char b_byte_R = byte_stream[ i * subframe_height + j ].z ;

            if( ( i == 0 ) && ( j == 0) ) {
                tile2[ (i * subframe_height + j)  ].x = ( b_byte_B ^ tile1[ (i * subframe_height + j) ].x ^ sd ) - b_byte_B ;
                tile2[ (i * subframe_height + j)  ].y = ( b_byte_B ^ tile1[ (i * subframe_height + j) ].y ^ sd ) - b_byte_B ;
                tile2[ (i * subframe_height + j)  ].z = ( b_byte_B ^ tile1[ (i * subframe_height + j) ].z ^ sd ) - b_byte_B ;
                //printf("(sd involved) output(%d,%d, c=%d) = (%d xor %d xor %d) - %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], sd, b_byte, output[ (y*width + x) * NUM_CHANNELS + c]);
            //} else if ( x == startCol ){
            //   output[ (y * width + x) * NUM_CHANNELS + c] = (b_byte ^  input[ ( y * width + x) * NUM_CHANNELS + c] ^  input[ ( ( y - 1 ) * width + ( endCol ) ) * NUM_CHANNELS + c ] ) - b_byte ; 
            } else {
                tile2[ (i * subframe_height + j) ].x = (b_byte_B ^  tile1[ ( i * subframe_height + j) ].x ^  tile1[ ( i* subframe_height + j )  - 1 ].x ) - b_byte_B ;
                tile2[ (i * subframe_height + j) ].y = (b_byte_B ^  tile1[ ( i * subframe_height + j) ].y ^  tile1[ ( i* subframe_height + j )  - 1 ].y ) - b_byte_B ;
                tile2[ (i * subframe_height + j) ].z = (b_byte_B ^  tile1[ ( i * subframe_height + j) ].z ^  tile1[ ( i* subframe_height + j )  - 1 ].z ) - b_byte_B ;
                                
                //printf("output(%d,%d, c= %d) = (%d xor %d xor %d) - %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], input[ ( y*width + (x ) )*NUM_CHANNELS + c -1],  b_byte, output[ (y*width + x) * NUM_CHANNELS + c] );
                //output[y*width+x] =255;
            }     
        } 
    }

    }

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_height + threadIdx.x) ].x ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_height + threadIdx.x) ].y ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_height + threadIdx.x) ].z ;    

}

__global__ void diffusionKernelv2( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array, int round ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_width + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    int index = ( gy * width + gx ) * NUM_CHANNELS ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    //if (gx >= width || gy >= height)
    //return ;

    //__shared__ unsigned char tile1[432] ;
    //__shared__ unsigned char tile2[432] ;
    //__shared__ uchar3 tile_byteStream[144] ; 
    //__shared__ unsigned char tile_byteStream[144] ;

    extern __shared__ unsigned char tilediffv2[] ;

    unsigned char *tile1 = tilediffv2 ;
    unsigned char *tile2 = tile1 + subframe_height * subframe_width * NUM_CHANNELS ;
    uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ;
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;

    //__shared__ uchar3 tile_byteStream[48] ;
    //unsigned char *tile_byteStream = tile2 + subframe_height * subframe_height * NUM_CHANNELS ;

    unsigned char safePixB =  (input)[ index + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ index + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ index + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ;

    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ (gy * width + gx) ] ;
    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

   // uchar3 safebyte = (&tile_byteStream[0])[gy * width + gx] ;


    /*if( blockId_x == 0 && blockId_y == 0 ){
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;  
    } */

    /*if ( blockId_x == 1 && blockId_y == 0 ){
        printf("\n") ;
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;
    } */

    /*unsigned char safebyte = byte_stream[ gy * width + gx ] ;
    tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safebyte ; */

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    //if (tid == 0 ){

    if( ( tid_x == 0 ) && ( tid_y == 0 ) ) {

    unsigned char sd ;

        if( round == 1 ) { //all rounds use the same set of sd values

            // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
            
            int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

            //printf("\n sd_block X : = %d \n", sd_block_x) ;
            int sd_block_y = (blockId_y + ((blockId_x + 1)/gridDim.x)) % gridDim.y ;

            //printf( "\n block (%d, %d) has sd block Y : = (%d, %d) \n ", blockId_x, blockId_y, sd_block_x, sd_block_y );
            
            /*int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                                  ( sd_block_x * subframe_width )    +
                                  ( gridDim.x * subframe_height * (subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;*/

            int sd_coordinate = (( sd_block_y * width * subframe_height ) + 
                                ( sd_block_x * subframe_width ) + 
                                ( subframe_height - 1 ) * width + subframe_width - 1) * NUM_CHANNELS ;
                
            //printf("\n sd coordinate: %d\n ", sd_coordinate );


            //printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
            sd = input[ sd_coordinate ] ;
            d_sd_array[ (blockId_y * gridDim.x) + blockId_x ] = sd ; 

            //sd = tile1[ 143 ];
        }
        
    //diffusionSeqv3( width, sd, tile1, tile_byteStream, tile2 , channelOffset , performInverseDiffusion, subframe_height ); //this is the sequential version based oon chebishev on the paper.. unfutunatly it is nonParallelizable
    
    int total = subframe_height * subframe_width ;

    //printf("total : %d ", total );
    
    unsigned char b_byte = tile_byteStream[ 0 ].x ;
    //unsigned char b_byte_G = tile_byteStream[ 0 ].y ;
    //unsigned char b_byte_R = tile_byteStream[ 0 ].z ;
    
    unsigned char in_B_component = tile1[ 0 ] ;
    unsigned char in_G_component = tile1[ 1 ] ;
    unsigned char in_R_component = tile1[ 2 ] ;
    
    tile2[ 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ sd ;
    tile2[ 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ sd ;
    tile2[ 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ sd ;


    for ( int  i = 1; i < total ; i++) {

        int base = i * 3;
        
        b_byte = tile_byteStream[ i ].x ;

        //b_byte = ((unsigned char*)&tile_byteStream[0])[i] ;

        /*if( blockId_x == 0 && blockId_y == 0 ){
            printf( "%d ", b_byte ) ;
        }*/

        /*if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
            printf(" %d ", b_byte ) ;   
        }*/
        

        in_B_component = tile1[ base + 0 ] ; 
        in_G_component = tile1[ base + 1 ] ; 
        in_R_component = tile1[ base + 2 ] ; 

        tile2[ base + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ tile2[ base - 1 ] ;
        tile2[ base + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ tile2[ base + 0 ] ;
        tile2[ base + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ tile2[ base + 1 ] ;                    

    }

    } 

    __syncthreads();

    output[ index + 0 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] ;
    output[ index + 1 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] ;
    output[ index + 2 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] ;

}

__global__ void invdiffusionKernelv2( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array ) {

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

    extern __shared__ unsigned char  tileinv2[] ;

    unsigned char *tile1 = tileinv2 ;
    unsigned char *tile2 = tileinv2 + subframe_height * subframe_width * NUM_CHANNELS ;
    uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ; 
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;
    //unsigned char *tile_byteStream = tile2 + subframe_height * subframe_height * NUM_CHANNELS ;
    //__shared__ unsigned char tile_byteStream[144] ;
    
    unsigned char safePixB =  (input)[ (gy * width + gx) * NUM_CHANNELS + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ (gy * width + gx) * NUM_CHANNELS + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ (gy * width + gx) * NUM_CHANNELS + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ; 

    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    /*unsigned char safebyte = (byte_stream)[ gy * width + gx ] ;
    tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safebyte ;*/

    unsigned char sd ;

    sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

    if( ( tid_x == 0 ) && ( tid_y == 0 ) ) {
        //invdiffusionSeqv2( width, sd, tile1, tile_byteStream, tile2 , channelOffset, performInverseDiffusion, subframe_height  ) ; 

        int total = subframe_height * subframe_width ;

        unsigned char b_byte_B = tile_byteStream[ 0 ].x ;
    
        unsigned char in_B_component = tile1[ 0 ] ;
        unsigned char in_G_component = tile1[ 1 ] ;
        unsigned char in_R_component = tile1[ 2 ] ;
    
        tile2[ 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ;
        tile2[ 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ;
        tile2[ 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

        for ( int i = 1 ; i < total ;  i ++ ) {

            int base = i * 3;
            
            unsigned char b_byte_B = tile_byteStream[ i ].x ;
            //unsigned char b_byte_B = ((unsigned char*)&tile_byteStream[0])[i] ;

            /*if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                printf(" %d ", b_byte_B ) ;   
            }*/

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

__global__ void diffusionKernelv4( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array, int round ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_width + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    int index = ( gy * width + gx ) * NUM_CHANNELS ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    extern __shared__ unsigned char tilediffv4[] ;

    //__shared__ unsigned char sd[8];

    unsigned char *tile1 = tilediffv4 ;
    unsigned char *tile2 = tile1 + subframe_height * subframe_width * NUM_CHANNELS ;
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

   // uchar3 safebyte = (&tile_byteStream[0])[gy * width + gx] ;


    /*if( blockId_x == 0 && blockId_y == 0 ){
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;  
    } */

    /*if ( blockId_x == 1 && blockId_y == 0 ){
        printf("\n") ;
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;
    } */

    /*unsigned char safebyte = byte_stream[ gy * width + gx ] ;
    tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safebyte ; */

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    //if (tid == 0 ){

    //if ( blockId_x == 0 && blockId_y == 0  ) {
    if(  tid_x == 0 ) {

        //if ( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {

            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ];

            if( round == 1 ) { //all rounds use the same set of sd values

                // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
            
                int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

                //printf("\n sd_block X : = %d \n", sd_block_x) ;
                int sd_block_y = ( blockId_y + ( (blockId_x + 1) / gridDim.x ) ) % gridDim.y ;

                //printf( "\n block (%d, %d) has sd block Y : = (%d, %d) \n ", blockId_x, blockId_y, sd_block_x, sd_block_y );
            
                /*int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                                  ( sd_block_x * subframe_width )    +
                                  ( gridDim.x * subframe_height * (subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;*/

                int sd_coordinate = (( sd_block_y * width * subframe_height ) + 
                                ( sd_block_x * subframe_width ) + 
                                ( subframe_height - 1 ) * width + subframe_width - 1) * NUM_CHANNELS ;
                
                //printf("\n sd coordinate: %d\n ", sd_coordinate );


                //printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
                sd = input[ sd_coordinate ] ;
                d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ] = sd ; 

                //printf("%d ", sd[tid_y]);

                //sd = tile1[ 143 ];
            }

       __syncthreads();
        
        int total = ( subframe_height * subframe_width ) / subframe_height ;

        //printf("total : %d \n", total );
    
        unsigned char b_byte = tile_byteStream[ tid_y * subframe_width ].x ;
        //unsigned char b_byte_G = tile_byteStream[ 0 ].y ;
        //unsigned char b_byte_R = tile_byteStream[ 0 ].z ;
    
        unsigned char in_B_component = tile1[ tid_y * subframe_width * 3 + 0 ] ;
        unsigned char in_G_component = tile1[ tid_y * subframe_width * 3 + 1 ] ;
        unsigned char in_R_component = tile1[ tid_y * subframe_width * 3 + 2 ] ;

        //if( tid_y == 0 ){
        //    for( int i = 0 ; i< 2; i++){
        //        for( int j = 0 ; j< 6; j++){
        //            printf(  "  %d  ", (int) tile1[i * 6 + j ]);
        //        }
        //    }printf("\n");
        //}
    
        tile2[ tid_y * subframe_width * 3 + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ sd ; 
        tile2[ tid_y * subframe_width * 3 + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ sd ; 
        tile2[ tid_y * subframe_width * 3 + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ sd ;

        //printf(" thread y id: %d, b_byte=%d xor (in_B_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
        //printf(" thread y id: %d, b_byte=%d xor (in_G_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
        //printf(" thread y id: %d, b_byte=%d xor (in_R_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

        __syncthreads();


        for ( int  i = tid_y * subframe_width + 1; i < (tid_y * subframe_width + total) ; i++) {

            int base = i * 3;
        
            b_byte = tile_byteStream[ i ].x ;

            //b_byte = ((unsigned char*)&tile_byteStream[0])[i] ;

            /*if( blockId_x == 0 && blockId_y == 0 ){
                printf( "%d ", b_byte ) ;
            }*/

            /*if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                printf(" %d ", b_byte ) ;   
            }*/

            in_B_component = tile1[ base + 0 ] ; 
            in_G_component = tile1[ base + 1 ] ; 
            in_R_component = tile1[ base + 2 ] ; 

            tile2[ base + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ tile2[ base - 1 ] ;
            tile2[ base + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ tile2[ base + 0 ] ;
            tile2[ base + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ tile2[ base + 1 ] ;                    

        }

        }
    //} 

    __syncthreads();

    output[ index + 0 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] ;
    output[ index + 1 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] ;
    output[ index + 2 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] ;
    //}

}

__global__ void invdiffusionKernelv4( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array ) {

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

    extern __shared__ unsigned char  tileinv4[] ;

    //__shared__ unsigned char sd[8] ;

    unsigned char *tile1 = tileinv4 ;
    unsigned char *tile2 = tileinv4 + subframe_height * subframe_width * NUM_CHANNELS ;
    uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ; 
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;
    //unsigned char *tile_byteStream = tile2 + subframe_height * subframe_height * NUM_CHANNELS ;
    //__shared__ unsigned char tile_byteStream[144] ;
    
    unsigned char safePixB =  (input)[ (gy * width + gx) * NUM_CHANNELS + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ (gy * width + gx) * NUM_CHANNELS + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ (gy * width + gx) * NUM_CHANNELS + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ; 

    uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    //unsigned char safebyte = (byte_stream)[ gy * width + gx ] ;
    //tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safebyte ;

    //if( blockId_x == 0 && blockId_y == 0 ){
    if( tid_x == 0  ) {

        //if( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {
        
            //invdiffusionSeqv2( width, sd, tile1, tile_byteStream, tile2 , channelOffset, performInverseDiffusion, subframe_height  ) ; 
            
            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

            //printf("%d ", sd[tid_y]);

            //__syncthreads();

            int total = (subframe_height * subframe_width) / subframe_height ;

            unsigned char b_byte_B = tile_byteStream[ tid_y * subframe_width ].x ;
    
            unsigned char in_B_component = tile1[ tid_y * subframe_width * 3 + 0 ] ;
            unsigned char in_G_component = tile1[ tid_y * subframe_width * 3 + 1 ] ;
            unsigned char in_R_component = tile1[ tid_y * subframe_width * 3 + 2 ] ;

            tile2[ tid_y * subframe_width * 3 + 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ; 
            tile2[ tid_y * subframe_width * 3 + 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ; 
            tile2[ tid_y * subframe_width * 3 + 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

            //printf(" \nthread y id: %d, (b_byte=%d xor in_B_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_G_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_R_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

            __syncthreads() ;

            for ( int i = tid_y * subframe_width + 1 ; i < (tid_y * subframe_width + total) ;  i ++ ) {

                int base = i * 3;
            
                unsigned char b_byte_B = tile_byteStream[ i ].x ;
                //unsigned char b_byte_B = ((unsigned char*)&tile_byteStream[0])[i] ;

                //if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                //    printf(" %d ", b_byte_B ) ;   
                //}

                tile2[ base + 0] = (b_byte_B ^  tile1[ base + 0 ] ^  tile1[ base - 1 ] ) - b_byte_B ;
                tile2[ base + 1] = (b_byte_B ^  tile1[ base + 1 ] ^  tile1[ base + 0 ] ) - b_byte_B ;
                tile2[ base + 2] = (b_byte_B ^  tile1[ base + 2 ] ^  tile1[ base + 1 ] ) - b_byte_B ;                     
            } 
        //}
    }

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2] ;    
    //}
}

__global__ void diffusionKernelv5( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array, int round ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_width + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    int index = ( gy * width + gx ) * NUM_CHANNELS ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    extern __shared__ unsigned char tilediffv4[] ;

    //__shared__ unsigned char sd[8];

    unsigned char *tile1 = tilediffv4 ;
    unsigned char *tile2 = tile1 + subframe_height * subframe_width * NUM_CHANNELS ;
    unsigned char *tile_byteStream = (tile2) + subframe_height * subframe_width * 3 ;
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;

    unsigned char safePixB =  (input)[ index + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ index + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ index + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ;

    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ (gy * width + gx) ] ;
    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    //tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    unsigned char safebyte = (byte_stream)[ globalIndex ];
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safebyte ;

   // uchar3 safebyte = (&tile_byteStream[0])[gy * width + gx] ;


    /*if( blockId_x == 0 && blockId_y == 0 ){
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;  
    } */

    /*if ( blockId_x == 1 && blockId_y == 0 ){
        printf("\n") ;
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;
    } */

    /*unsigned char safebyte = byte_stream[ gy * width + gx ] ;
    tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safebyte ; */

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    //if (tid == 0 ){

    //if ( blockId_x == 0 && blockId_y == 0  ) {
    if(  tid_x == 0 || tid_x == (subframe_width/2 ) ) {

        //if ( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {

            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ];

            if( round == 1 ) { //all rounds use the same set of sd values

                // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
            
                int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

                //printf("\n sd_block X : = %d \n", sd_block_x) ;
                int sd_block_y = ( blockId_y + ( (blockId_x + 1) / gridDim.x ) ) % gridDim.y ;

                //printf( "\n block (%d, %d) has sd block Y : = (%d, %d) \n ", blockId_x, blockId_y, sd_block_x, sd_block_y );
            
                /*int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                                  ( sd_block_x * subframe_width )    +
                                  ( gridDim.x * subframe_height * (subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;*/

                int sd_coordinate = (( sd_block_y * width * subframe_height ) + 
                                ( sd_block_x * subframe_width ) + 
                                ( subframe_height - 1 ) * width + subframe_width - 1) * NUM_CHANNELS ;
                
                //printf("\n sd coordinate: %d\n ", sd_coordinate );


                //printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
                sd = input[ sd_coordinate ] ;
                d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ] = sd ; 

                //printf("%d ", sd[tid_y]);

                //sd = tile1[ 143 ];
            }

       __syncthreads();
        
        int total = ( ( subframe_height * subframe_width ) / subframe_height ) / 2 ;

        //printf("total : %d \n", total );
    
        unsigned char b_byte = tile_byteStream[ tid_y * subframe_width + tid_x ] ;
        //unsigned char b_byte_G = tile_byteStream[ 0 ].y ;
        //unsigned char b_byte_R = tile_byteStream[ 0 ].z ;
    
        unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
        unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
        unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;

        //if( tid_y == 0 ){
        //    for( int i = 0 ; i< 2; i++){
        //        for( int j = 0 ; j< 6; j++){
        //            printf(  "  %d  ", (int) tile1[i * 6 + j ]);
        //        }
        //    }printf("\n");
        //}
    
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ sd ; 
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ sd ; 
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ sd ;

        //printf(" thread y id: %d, b_byte=%d xor (in_B_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
        //printf(" thread y id: %d, b_byte=%d xor (in_G_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
        //printf(" thread y id: %d, b_byte=%d xor (in_R_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

        __syncthreads();


        for ( int  i = tid_y * subframe_width + tid_x + 1; i < (tid_y * subframe_width + tid_x + total) ; i++) {

            int base = i * 3;
        
            b_byte = tile_byteStream[ i ] ;

            //b_byte = ((unsigned char*)&tile_byteStream[0])[i] ;

            /*if( blockId_x == 0 && blockId_y == 0 ){
                printf( "%d ", b_byte ) ;
            }*/

            /*if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                printf(" %d ", b_byte ) ;   
            }*/

            in_B_component = tile1[ base + 0 ] ; 
            in_G_component = tile1[ base + 1 ] ; 
            in_R_component = tile1[ base + 2 ] ; 

            tile2[ base + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ tile2[ base - 1 ] ;
            tile2[ base + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ tile2[ base + 0 ] ;
            tile2[ base + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ tile2[ base + 1 ] ;                    

        }

        }
    //} 

    __syncthreads();

    output[ index + 0 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] ;
    output[ index + 1 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] ;
    output[ index + 2 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] ;
    //}

}

__global__ void invdiffusionKernelv5( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array ) {

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

    extern __shared__ unsigned char  tileinv4[] ;

    //__shared__ unsigned char sd[8] ;

    unsigned char *tile1 = tileinv4 ;
    unsigned char *tile2 = tileinv4 + subframe_height * subframe_width * NUM_CHANNELS ;
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ; 
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;
    
    unsigned char *tile_byteStream = tile2 + subframe_height * subframe_width * NUM_CHANNELS;
    //__shared__ unsigned char tile_byteStream[144] ;
    
    unsigned char safePixB =  (input)[ (gy * width + gx) * NUM_CHANNELS + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ (gy * width + gx) * NUM_CHANNELS + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ (gy * width + gx) * NUM_CHANNELS + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ; 

    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    //tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    unsigned char safebyte = (byte_stream)[ globalIndex ] ;
    tile_byteStream[ threadIdx.y * subframe_width + threadIdx.x ] = safebyte ;

    //if( blockId_x == 0 && blockId_y == 0 ){
    if( tid_x == 0 || tid_x == (subframe_width/2 )  ) {

        //if( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {
        
            //invdiffusionSeqv2( width, sd, tile1, tile_byteStream, tile2 , channelOffset, performInverseDiffusion, subframe_height  ) ; 
            
            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

            //printf("%d ", sd[tid_y]);

            //__syncthreads();

            int total = ( (subframe_height * subframe_width) / subframe_height ) / 2 ;

            unsigned char b_byte_B = tile_byteStream[ tid_y * subframe_width + tid_x ] ; 
    
            unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
            unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
            unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;

            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

            //printf(" \nthread y id: %d, (b_byte=%d xor in_B_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_G_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_R_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

            __syncthreads() ;

            for ( int i = tid_y * subframe_width + tid_x + 1 ; i < (tid_y * subframe_width + tid_x + total) ;  i ++ ) {

                int base = i * 3 ;
            
                unsigned char b_byte_B = tile_byteStream[ i ] ;
                //unsigned char b_byte_B = ((unsigned char*)&tile_byteStream[0])[i] ;

                //if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                //    printf(" %d ", b_byte_B ) ;   
                //}

                tile2[ base + 0] = (b_byte_B ^  tile1[ base + 0 ] ^  tile1[ base - 1 ] ) - b_byte_B ;
                tile2[ base + 1] = (b_byte_B ^  tile1[ base + 1 ] ^  tile1[ base + 0 ] ) - b_byte_B ;
                tile2[ base + 2] = (b_byte_B ^  tile1[ base + 2 ] ^  tile1[ base + 1 ] ) - b_byte_B ;                     
            } 
        }
    //}

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2] ;    
    //}
}

__global__ void diffusionKernelv6( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array, int round ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;
    //int channelOffset = blockId_x / gridDim.x ;

    int gx = blockIdx.x * subframe_width + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    int index = ( gy * width + gx ) * NUM_CHANNELS ;

    int blockId = blockIdx.y * gridDim.x + blockIdx.x ;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x ;
    int globalIndex = blockId * ( blockDim.x * blockDim.y ) + threadId ;

    extern __shared__ unsigned char tilediffv6[] ;

    //__shared__ unsigned char sd[8];

    unsigned char *tile1 = tilediffv6 ;
    unsigned char *tile2 = tile1 + subframe_height * subframe_width * NUM_CHANNELS ;
    unsigned char *tile_byteStream = (tile2) + subframe_height * subframe_width * 3 ;
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;

    unsigned char safePixB =  (input)[ index + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ index + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ index + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ;

    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ (gy * width + gx) ] ;
    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    //tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    unsigned char safebyte = (byte_stream)[ globalIndex ];
    tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safebyte ;

   // uchar3 safebyte = (&tile_byteStream[0])[gy * width + gx] ;


    /*if( blockId_x == 0 && blockId_y == 0 ){
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;  
    } */

    /*if ( blockId_x == 1 && blockId_y == 0 ){
        printf("\n") ;
        printf( "\n%d, %d, %d ", safe3bytes.x, safe3bytes.y, safe3bytes.z ) ;
    } */

    /*unsigned char safebyte = byte_stream[ gy * width + gx ] ;
    tile_byteStream[ threadIdx.y * subframe_height + threadIdx.x ] = safebyte ; */

    //channelOffset = 0 ; //focusing on the blue channel for now just for debugging purposes ....

    //__shared__ unsigned char diffused_subframe[TOTALPIXELSSUBFRAME]; // allocating shared memory

    //if (tid == 0 ){

    //if ( blockId_x == 0 && blockId_y == 0  ) {
    if(  tid_x == 0 || (!(tid_x & 2)) ) {

        //if ( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {

            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ];

            if( round == 1 ) { //all rounds use the same set of sd values

                // the diffusion seed of a specific subframe i is the last pixel value of the subframe( (i+1) mod n ) (n being number of assitant threads in the article (n being nummber of blocs in my case ))
            
                int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 

                //printf("\n sd_block X : = %d \n", sd_block_x) ;
                int sd_block_y = ( blockId_y + ( (blockId_x + 1) / gridDim.x ) ) % gridDim.y ;

                //printf( "\n block (%d, %d) has sd block Y : = (%d, %d) \n ", blockId_x, blockId_y, sd_block_x, sd_block_y );
            
                /*int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                                  ( sd_block_x * subframe_width )    +
                                  ( gridDim.x * subframe_height * (subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;*/

                int sd_coordinate = (( sd_block_y * width * subframe_height ) + 
                                ( sd_block_x * subframe_width ) + 
                                ( subframe_height - 1 ) * width + subframe_width - 1) * NUM_CHANNELS ;
                
                //printf("\n sd coordinate: %d\n ", sd_coordinate );


                //printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
                sd = input[ sd_coordinate ] ;
                d_sd_array[ (blockId_y * gridDim.x) + blockId_x  ] = sd ; 

                //printf("%d ", sd[tid_y]);

                //sd = tile1[ 143 ];
            }

       __syncthreads();
        
        int total = 4 ;

        //printf("total : %d \n", total );
    
        unsigned char b_byte = tile_byteStream[ tid_y * subframe_width + tid_x ] ;
        //unsigned char b_byte_G = tile_byteStream[ 0 ].y ;
        //unsigned char b_byte_R = tile_byteStream[ 0 ].z ;
    
        unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
        unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
        unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;

        //if( tid_y == 0 ){
        //    for( int i = 0 ; i< 2; i++){
        //        for( int j = 0 ; j< 6; j++){
        //            printf(  "  %d  ", (int) tile1[i * 6 + j ]);
        //        }
        //    }printf("\n");
        //}
    
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ sd ; 
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ sd ; 
        tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ sd ;

        //printf(" thread y id: %d, b_byte=%d xor (in_B_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
        //printf(" thread y id: %d, b_byte=%d xor (in_G_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
        //printf(" thread y id: %d, b_byte=%d xor (in_R_component= %d + b_byte) xor sd=%d = %d \n", tid_y, b_byte, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

        __syncthreads();


        for ( int  i = tid_y * subframe_width + tid_x + 1; i < (tid_y * subframe_width + tid_x + total) ; i++) {

            int base = i * 3;
        
            b_byte = tile_byteStream[ i ] ;

            //b_byte = ((unsigned char*)&tile_byteStream[0])[i] ;

            /*if( blockId_x == 0 && blockId_y == 0 ){
                printf( "%d ", b_byte ) ;
            }*/

            /*if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                printf(" %d ", b_byte ) ;   
            }*/

            in_B_component = tile1[ base + 0 ] ; 
            in_G_component = tile1[ base + 1 ] ; 
            in_R_component = tile1[ base + 2 ] ; 

            tile2[ base + 0 ] = b_byte ^ ( in_B_component + b_byte ) ^ tile2[ base - 1 ] ;
            tile2[ base + 1 ] = b_byte ^ ( in_G_component + b_byte ) ^ tile2[ base + 0 ] ;
            tile2[ base + 2 ] = b_byte ^ ( in_R_component + b_byte ) ^ tile2[ base + 1 ] ;                    

        }

        }
    //} 

    __syncthreads();

    output[ index + 0 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] ;
    output[ index + 1 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] ;
    output[ index + 2 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] ;
    //}

}

__global__ void invdiffusionKernelv6( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion, unsigned char * d_sd_array ) {

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

    extern __shared__ unsigned char  tileinv4[] ;

    //__shared__ unsigned char sd[8] ;

    unsigned char *tile1 = tileinv4 ;
    unsigned char *tile2 = tileinv4 + subframe_height * subframe_width * NUM_CHANNELS ;
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + subframe_height * subframe_width ; 
    //uchar3 *tile_byteStream = reinterpret_cast<uchar3*>(tile2) + 48 ;
    
    unsigned char *tile_byteStream = tile2 + subframe_height * subframe_width * NUM_CHANNELS;
    //__shared__ unsigned char tile_byteStream[144] ;
    
    unsigned char safePixB =  (input)[ (gy * width + gx) * NUM_CHANNELS + 0 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] = safePixB ; 

    unsigned char safePixG =  (input)[ (gy * width + gx) * NUM_CHANNELS + 1 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] = safePixG ;

    unsigned char safePixR =  (input)[ (gy * width + gx) * NUM_CHANNELS + 2 ] ;
    tile1[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] = safePixR ; 

    //uchar3 safe3bytes = reinterpret_cast<uchar3*>(byte_stream)[ globalIndex ] ;
    //tile_byteStream[  threadIdx.y * subframe_width + threadIdx.x ] = safe3bytes ; 

    unsigned char safebyte = (byte_stream)[ globalIndex ] ;
    tile_byteStream[ threadIdx.y * subframe_width + threadIdx.x ] = safebyte ;

    //if( blockId_x == 0 && blockId_y == 0 ){
    if( tid_x == 0 || tid_x == (subframe_width/2 )  ) {

        //if( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {
        
            //invdiffusionSeqv2( width, sd, tile1, tile_byteStream, tile2 , channelOffset, performInverseDiffusion, subframe_height  ) ; 
            
            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

            //printf("%d ", sd[tid_y]);

            //__syncthreads();

            int total = ( (subframe_height * subframe_width) / subframe_height ) / 2 ;

            unsigned char b_byte_B = tile_byteStream[ tid_y * subframe_width + tid_x ] ; 
    
            unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
            unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
            unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;

            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

            //printf(" \nthread y id: %d, (b_byte=%d xor in_B_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_G_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_R_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

            __syncthreads() ;

            for ( int i = tid_y * subframe_width + tid_x + 1 ; i < (tid_y * subframe_width + tid_x + total) ;  i ++ ) {

                int base = i * 3 ;
            
                unsigned char b_byte_B = tile_byteStream[ i ] ;
                //unsigned char b_byte_B = ((unsigned char*)&tile_byteStream[0])[i] ;

                //if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                //    printf(" %d ", b_byte_B ) ;   
                //}

                tile2[ base + 0] = (b_byte_B ^  tile1[ base + 0 ] ^  tile1[ base - 1 ] ) - b_byte_B ;
                tile2[ base + 1] = (b_byte_B ^  tile1[ base + 1 ] ^  tile1[ base + 0 ] ) - b_byte_B ;
                tile2[ base + 2] = (b_byte_B ^  tile1[ base + 2 ] ^  tile1[ base + 1 ] ) - b_byte_B ;                     
            } 
        }
    //}

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2] ;    
    //}
}
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

    //unsigned char safebyte = (byte_stream)[ globalIndex ] ;

    //printf(" block(%d, %d) , thread (%d, %d) prende safebyte %d \n", blockId_x, blockId_y, tid_x, tid_y, safebyte ) ;

    //tile_byteStream[ threadIdx.y * subframe_width + threadIdx.x ] = safebyte ;

    //if(blockId_x == 2 && blockId_y == 2) {
    //if( blockId_x == 0 && blockId_y == 0 ){
    if (  tid_x == 0 || ( !(tid_x & 1) ) ) {

        //if( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {
        
            //invdiffusionSeqv2( width, sd, tile1, tile_byteStream, tile2 , channelOffset, performInverseDiffusion, subframe_height  ) ; 
            
            unsigned char sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

            //printf("%d ", sd[tid_y]);

            //__syncthreads();

            int total = 2 ;

            unsigned char b_byte_B = tile_byteStream[ tid_y * subframe_width + tid_x ].x ; 
    
            unsigned char in_B_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] ;
            unsigned char in_G_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] ;
            unsigned char in_R_component = tile1[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] ;

            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 0 ] = ( b_byte_B ^ in_B_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 1 ] = ( b_byte_B ^ in_G_component ^ sd ) - b_byte_B ; 
            tile2[ ( tid_y * subframe_width + tid_x ) * 3 + 2 ] = ( b_byte_B ^ in_R_component ^ sd ) - b_byte_B ;

            //printf(" \nthread y id: %d, (b_byte=%d xor in_B_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_B_component, sd, tile2[ tid_y * subframe_width + 0 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_G_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_G_component, sd, tile2[ tid_y * subframe_width + 1 ] );
            //printf(" \nthread y id: %d, (b_byte=%d xor in_R_component= %d xor sd=%d - b_byte = %d)", tid_y, b_byte_B, in_R_component, sd, tile2[ tid_y * subframe_width + 2 ] );

            __syncthreads() ;

            for ( int i = tid_y * subframe_width + tid_x + 1 ; i < (tid_y * subframe_width + tid_x + total) ;  i ++ ) {

                int base = i * 3 ;
            
                unsigned char b_byte_B = tile_byteStream[ i ].x ;

                //printf(" %d \n", blockId_x, blockId_y, tid_x, tid_y, b_byte_B ) ;

                //unsigned char b_byte_B = ((unsigned char*)&tile_byteStream[0])[i] ;

                //if( blockId_x == 0 && blockId_y == 0){ // debugging for block (0,0)
                //    printf(" %d ", b_byte_B ) ;   
                //}

                tile2[ base + 0] = (b_byte_B ^  tile1[ base + 0 ] ^  tile1[ base - 1 ] ) - b_byte_B ;
                tile2[ base + 1] = (b_byte_B ^  tile1[ base + 1 ] ^  tile1[ base + 0 ] ) - b_byte_B ;
                tile2[ base + 2] = (b_byte_B ^  tile1[ base + 2 ] ^  tile1[ base + 1 ] ) - b_byte_B ;                     
            } 
        }
    //}

    __syncthreads();

    output[ (gy * width + gx) * NUM_CHANNELS + 0  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 1  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1] ;
    output[ (gy * width + gx) * NUM_CHANNELS + 2  ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2] ;    
    //}
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

        //if ( tid_y == 0 || tid_y == 2 || tid_y == 4 || tid_y == 6 ) {

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
        //unsigned char b_byte_G = tile_byteStream[ 0 ].y ;
        //unsigned char b_byte_R = tile_byteStream[ 0 ].z ;

        //unsigned char b_byte = tile_byteStream[ tid_y * subframe_width + tid_x ] ;

        //printf(" block(%d, %d) , thread (%d, %d) prende safebyte %d \n", blockId_x, blockId_y, tid_x, tid_y, b_byte ) ;
    
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

        //b_byte = tile_byteStream[ i ] ;

        //printf(" block(%d, %d) , thread (%d, %d) prende safebyte %d \n", blockId_x, blockId_y, tid_x, tid_y, b_byte ) ;

        in_B_component = tile1[ base + 0 ] ; 
        in_G_component = tile1[ base + 1 ] ; 
        in_R_component = tile1[ base + 2 ] ; 

        tile2[ base + 0 ] = b_byte_B ^ ( in_B_component + b_byte_B ) ^ tile2[ base - 1 ] ;
        tile2[ base + 1 ] = b_byte_B ^ ( in_G_component + b_byte_B ) ^ tile2[ base + 0 ] ;
        tile2[ base + 2 ] = b_byte_B ^ ( in_R_component + b_byte_B ) ^ tile2[ base + 1 ] ;                    

        }
    //} 

    __syncthreads();

    output[ index + 0 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 0 ] ;
    output[ index + 1 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 1 ] ;
    output[ index + 2 ] =  tile2[ (threadIdx.y * subframe_width + threadIdx.x) * NUM_CHANNELS + 2 ] ;
   
    //}

}

void diffusionOpWrapper( unsigned char *d_byteStream, unsigned char * d_input, unsigned char *input, unsigned char * d_output, unsigned char *output, int width, int height, int subframe_height, int subframe_width, int performInverseDiffusion,  unsigned char *d_sd_array, cudaStream_t stream1 ) {

    const int total_pixels = width * height ;  // 921600, 960x960
    const int num_subframesXdir = width / subframe_width ; // 80, 960x960
    const int num_subframesYdir = height / subframe_height ; // 120, 960x960

    static int round = 0 ;

    ( performInverseDiffusion == 0 ) ? round ++ : round -- ;

    //cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyHostToDevice) ;

    //if( performInverseDiffusion == 1 && round == (ROUNDS - 1) ) {
        /*printf("sd values used for inverse diffusion\n") ;
        for(int i = 0; i < sd_array.size(); i ++){
            printf("% d", sd_array[i]); 
        }*/
    //    cudaMemcpy( d_sd_array, sd_array.data(), num_subframesXdir * num_subframesYdir * sizeof( unsigned char ) , cudaMemcpyHostToDevice ) ;
    //}

    dim3 blocksPerGrid( num_subframesXdir , num_subframesYdir ) ;
    //dim3 threadsPerBlock(154,6) ; // maximum number of threads per block that can be allocated. // same way as confusion
    dim3 threadsPerBlock( subframe_width , subframe_height ) ;

    size_t sharedMemPerBlock = 3 * subframe_height * subframe_width * NUM_CHANNELS * sizeof ( unsigned char ) ;
    //size_t sharedMemPerBlock =  ( 2 * subframe_height * subframe_width * NUM_CHANNELS * sizeof( unsigned char ) ) + ( subframe_height * subframe_width * sizeof( unsigned char) );

    if( performInverseDiffusion == 0 ) { // launch diffusion kernel
        //printf("executing diffusion Kernel...\n") ;
        diffusionKernelv7 <<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock, stream1 >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, subframe_width, performInverseDiffusion, d_sd_array, round );
    //cudaDeviceSynchronize() ;
    } else { // launch inverse diffusion kernel 
        invdiffusionKernelv7 <<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock, stream1 >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, subframe_width, performInverseDiffusion, d_sd_array );
    //cudaDeviceSynchronize() ;
    } 
    
    //cudaMemcpy( output, d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyDeviceToHost );
    
    //if( performInverseDiffusion == 0  && round == 1 ) {
        //sd_array.resize( num_subframesXdir * num_subframesYdir ) ;
        //cudaMemcpy( sd_array.data(), d_sd_array,  num_subframesXdir * num_subframesYdir * sizeof(unsigned char) , cudaMemcpyDeviceToHost ) ;
        //printf("sd array size: %d ", sd_array.size());
        //for(int i = 0; i < sd_array.size(); i ++){
        //    printf(" %d ", sd_array[i]);
        //} 
    //}
    
    //if(round == 0) {
    //    cudaFree(d_sd_array) ;
    //}
}

__global__ void coalescedLoadKernel(unsigned char * input, unsigned char * output, int width, int height) {

    extern __shared__ unsigned char tile[];

    int gx = blockIdx.x * blockDim.x + threadIdx.x ;
    int gy = blockIdx.y * blockDim.y + threadIdx.y ;

    if(gx >= width || gy >= height )
        return ;
    
    int globalIdx = gy * width + gx ;
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x ;

    //coalesced global -> shared
    tile[localIdx] = input[globalIdx] ;

    __syncthreads();

    output[globalIdx] = tile[localIdx];

}

void runCoalescedLoadWrapper( unsigned char *input, unsigned char *output , int width, int height ) {

    unsigned char * d_input ;
    unsigned char * d_output ;

    cudaMalloc(&d_input, width * height * sizeof( unsigned char  )) ;
    cudaMalloc(&d_output, width * height * sizeof( unsigned char )) ;

    cudaMemcpy(d_input, (input), width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );

    dim3 block(12, 12); //256 threads per block
    dim3 grid( (width + block.x - 1)/block.x, (height + block.y - 1)/block.y ) ;

    size_t shmem_size = block.x * block.y * 3 * sizeof( unsigned char  ) ;
    coalescedLoadKernel <<< grid, block, shmem_size>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize() ;

    cv::Mat result(height, width, CV_8UC4);
    cudaMemcpy( output, d_output, width * height * sizeof( unsigned char ), cudaMemcpyDeviceToHost );
    cudaFree(d_input) ;
    cudaFree(d_output) ;

}