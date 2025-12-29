#include <iostream>
#include <stdint.h>
#include "../include/diffusion_kernel.hpp"

#define NUM_CHANNELS 3

//#define TOTALPIXELSSUBFRAME 4608 //the specifc subframe at focus is 6 * 768 pixels in total

/*__device__ void diffusionSeq(int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, unsigned char *input, const unsigned char *byte_stream, unsigned char *output, int performInverseDiffusion ){
    
    unsigned char b_byte ;

    //printf("DIFFUSION RESULTS: \n");
    for ( int y = startRow , i = 0; y <= endRow ; y ++, i++ ) {
        for( int x = startCol , j = 0 ; x <= endCol ; x ++, j ++ ){
            for(int c = 0; c < NUM_CHANNELS; c ++ ) {
                b_byte = byte_stream[ ( y * width + x ) * NUM_CHANNELS + c] ;
                if( (y == startRow) && (x == startCol) ) {
                    output[ (y*width+x)*NUM_CHANNELS + c] = b_byte ^ ( input[ (y * width + x) * NUM_CHANNELS + c] + b_byte ) ^ sd ;
                //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y,c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, sd, output[ (y*width + x) * NUM_CHANNELS + c]);
                } else if ( x == startCol ) {
                    output[ (y*width+x)*NUM_CHANNELS + c] = b_byte ^ ( input[ (y * width + x) * NUM_CHANNELS + c] + b_byte ) ^  output[ ( (y-1)*width + (endCol) )*NUM_CHANNELS + c] ;   
                }
                else {
                    output[ (y*width+x)*NUM_CHANNELS + c] = b_byte ^ ( input[ (y * width + x) * NUM_CHANNELS + c] + b_byte ) ^ output[ (y*width + (x) )*NUM_CHANNELS + c -1] ;
                    //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
                }
                //output[ (y*width + x)*NUM_CHANNELS + c ] = 100;    
            }
        //output[ (y*width + x) * NUM_CHANNELS + 0  ] = input[ i * 32 + j ].x ;  
        //output[ (y*width + x) * NUM_CHANNELS + 1  ] = input[ i * 32 + j ].y ;  
        //output[ (y*width + x) * NUM_CHANNELS + 2  ] = input[ i * 32 + j ].z ;      
        } 
    } 
} */

__device__ void diffusionSeq(int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, uchar3 *tileIn, uchar3 *tileOut, uchar3 *tile_byte_stream, int performInverseDiffusion ){
    
    unsigned char b_byte_x ;
    unsigned char b_byte_y ;
    unsigned char b_byte_z ;

    //printf("DIFFUSION RESULTS: \n");
    for ( int y = startRow , i = 0; y <= endRow ; y ++, i++ ) {
        for( int x = startCol , j = 0 ; x <= endCol ; x ++, j ++ ) {
            //for(int c = 0; c < NUM_CHANNELS; c ++ ) {
                b_byte_x = tile_byte_stream[ i * 32 + j ].x ;
                b_byte_y = tile_byte_stream[ i * 32 + j ].y ;
                b_byte_z = tile_byte_stream[ i * 32 + j ].z ;

                if( (y == startRow) && (x == startCol) ) {
                    tileOut[ (i * 32 + j ) ].x = b_byte_x ^ ( tileIn[ (i * 32 + j) ].x + b_byte_x ) ^ sd ;
                    tileOut[ (i * 32 + j ) ].y = b_byte_y ^ ( tileIn[ (i * 32 + j) ].y + b_byte_y ) ^ sd ;
                    tileOut[ (i * 32 + j ) ].z = b_byte_z ^ ( tileIn[ (i * 32 + j) ].z + b_byte_z ) ^ sd ;

                //printf("(sd involved) output(%d,%d, c=%d) = %d xor (%d + %d) xor %d = %d\n", x, y, 0, b_byte_x, input[ y * width + x].x, b_byte_x, sd, output[ (y*width + x) * NUM_CHANNELS + 0]);
                } else if ( x == startCol ) {
                    tileOut[ (i * 32 + j ) ].x = b_byte_x ^ ( tileIn[ (i * 32 + j) ].x + b_byte_x ) ^  tileOut[ ( (i - 1) * 32 + (31) ) ].x ;   
                    tileOut[ (i * 32 + j ) ].y = b_byte_y ^ ( tileIn[ (i * 32 + j) ].y + b_byte_y ) ^  tileOut[ ( (i - 1) * 32 + (31) ) ].y ;   
                    tileOut[ (i * 32 + j ) ].z = b_byte_z ^ ( tileIn[ (i * 32 + j) ].z + b_byte_z ) ^  tileOut[ ( (i - 1) * 32 + (31) ) ].z ;   
                
                }
                else {
                    tileOut[ (i * 32 + j ) ].x = b_byte_x ^ ( tileIn[ (i * 32 + j) ].x + b_byte_x ) ^ tileOut[ (i * 32 + (j) - 1) ].x ;
                    tileOut[ (i * 32 + j ) ].y = b_byte_y ^ ( tileIn[ (i * 32 + j) ].y + b_byte_y ) ^ tileOut[ (i * 32 + (j) - 1) ].y ;
                    tileOut[ (i * 32 + j ) ].z = b_byte_z ^ ( tileIn[ (i * 32 + j) ].z + b_byte_z ) ^ tileOut[ (i * 32 + (j) - 1) ].z ;
                    
                    //printf("output(%d,%d, c= %d) = %d xor (%d + %d) xor %d = %d\n", x, y, c, b_byte, input[ (y*width + x) * NUM_CHANNELS + c], b_byte, output[ (y*width + x )*NUM_CHANNELS + c -1], output[ (y*width+x)*NUM_CHANNELS + c]);                    
                }
                //output[ (y*width + x)*NUM_CHANNELS + c ] = 100;    
            //}    
        } 
    } 
}

/*__global__ void inverseDiffusionKernel(  unsigned char *input, unsigned char *output, const unsigned char *byte_stream, int width, int height, int subframe_height, int num_subframes, int performInverseDiffusion, unsigned char * d_sd_array ){
    
    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid = threadIdx.x;
    int channelOffset = blockId_x / gridDim.x ;

    unsigned char sd;

    int startRow =  blockId_y  * subframe_height ;  //startRow of the subframe
    int endRow =  startRow + subframe_height - 1 ;

    int startCol = blockId_x * subframe_height ;
    int endCol = startCol + subframe_height - 1 ;
   //printf ( "startRow of blockId %d: %d \n", blockId, startRow) ;

    sd = d_sd_array[ (blockId_y * gridDim.x ) + blockId_x ] ;

    diffusionSeq(startRow, endRow , startCol, endCol, width, sd, input, byte_stream, output , performInverseDiffusion  );
}  */

// device function: faster, avoids repeated multiplications, uses registers
__device__ __forceinline__ void diffusionSeq_fast( int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, uchar3 *tileIn, uchar3 *tileOut, uchar3 *tile_byte_stream, int performInverseDiffusion)
{
    // compute local tile height (assumes square tile)
    const int H = endRow - startRow + 1;

    int idx = 0;                    // linear index inside the tile: 0 .. H*H-1
    uchar3 prevOut = make_uchar3(0,0,0);

    // iterate in row-major order using local indices
    for (int i = 0; i < H; ++i) {
        // you could compute row_base = i*H but we use idx and increment
        for (int j = 0; j < H; ++j, ++idx) {
            // load inputs into registers (shared mem loads)
            uchar3 bs = tile_byte_stream[idx];   // b_byte_x,y,z
            uchar3 in  = tileIn[idx];

            uchar3 out;

            if (idx == 0) {
                // first element uses sd
                out.x = (unsigned char)( bs.x ^ (unsigned char)(in.x + bs.x) ^ sd );
                out.y = (unsigned char)( bs.y ^ (unsigned char)(in.y + bs.y) ^ sd );
                out.z = (unsigned char)( bs.z ^ (unsigned char)(in.z + bs.z) ^ sd );
            } else {
                // depends on previously computed out (prevOut)
                out.x = (unsigned char)( bs.x ^ (unsigned char)(in.x + bs.x) ^ prevOut.x );
                out.y = (unsigned char)( bs.y ^ (unsigned char)(in.y + bs.y) ^ prevOut.y );
                out.z = (unsigned char)( bs.z ^ (unsigned char)(in.z + bs.z) ^ prevOut.z );
            }

            // single shared-memory write
            tileOut[idx] = out;

            // update prevOut in registers
            prevOut = out;
        }
    }
}

__device__ __forceinline__ void diffusionSeq_fast_u4( int startRow, int endRow, int startCol, int endCol, int width, unsigned char sd, uchar4 *tileIn, uchar4 *tileOut, uchar4 *tile_byte_stream, int performInverseDiffusion ) {
    // compute local tile height (assumes square tile)
    const int H = endRow - startRow + 1;

    int idx = 0;                    // linear index inside the tile: 0 .. H*H-1
    uchar4 prevOut = make_uchar4(0, 0, 0, 0);

    // iterate in row-major order using local indices
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j, ++idx) {
            // load inputs from shared memory
            uchar4 bs = tile_byte_stream[idx];   // byte stream
            uchar4 in = tileIn[idx];             // input pixel

            uchar4 out;

            if (idx == 0) {
                // first element uses sd
                out.x = (unsigned char)(bs.x ^ (in.x + bs.x) ^ sd);
                out.y = (unsigned char)(bs.y ^ (in.y + bs.y) ^ sd);
                out.z = (unsigned char)(bs.z ^ (in.z + bs.z) ^ sd);
                out.w = 0; // unused
            } else {
                // depends on previously computed out (prevOut)
                out.x = (unsigned char)(bs.x ^ (in.x + bs.x) ^ prevOut.x);
                out.y = (unsigned char)(bs.y ^ (in.y + bs.y) ^ prevOut.y);
                out.z = (unsigned char)(bs.z ^ (in.z + bs.z) ^ prevOut.z);
                out.w = 0; // unused
            }

            // write result back to shared memory
            tileOut[idx] = out;

            // update prevOut in registers
            prevOut = out;
        }
    }
}

__global__ void diffusionKernel( unsigned char *input, unsigned char *output, unsigned char *byte_stream, int width, int height, int subframe_height, int num_subframes, int performInverseDiffusion, unsigned char * d_sd_array ) {

    int blockId_x = blockIdx.x ;
    int blockId_y = blockIdx.y ;
    int tid_x = threadIdx.x ;
    int tid_y = threadIdx.y ;

    int gx = blockIdx.x * subframe_height + threadIdx.x ;
    int gy = blockIdx.y * subframe_height + threadIdx.y ;

    if (gx >= width || gy >= height)
    return;

    extern __shared__ uchar4 tile[] ;

    uchar4 *tile1 = tile ;
    uchar4 *tile2 = tile + subframe_height * subframe_height ;
    uchar4 *tile_byteStream = tile2 + subframe_height * subframe_height ;
  
        uchar3 safePix = reinterpret_cast<uchar3*>(input)[gy * width + gx] ;    
        tile1[ ( threadIdx.y * subframe_height + threadIdx.x) ] = make_uchar4(safePix.x, safePix.y, safePix.z, 0) ;

        //__syncthreads() ;

        uchar3 safe3Bytes = reinterpret_cast<uchar3*>(byte_stream)[gy * width + gx] ;
        tile_byteStream[ ( threadIdx.y * subframe_height + threadIdx.x ) ] = make_uchar4(safe3Bytes.x, safe3Bytes.y, safe3Bytes.z, 0) ;

     if( ( tid_x == 0 ) && ( tid_y == 0 ) ) {

        unsigned char sd;

        int startRow =  blockId_y  * subframe_height ;  //startRow of the subframe
        int endRow = startRow + subframe_height - 1 ;
        int startCol =  blockId_x * subframe_height ;
        int endCol = startCol + subframe_height - 1 ;
        int sd_block_x = (blockId_x + 1) % gridDim.x ; //this is the y_coordinate of the selected pixel if we imagine the frame values in 2d space!!! if want to get the position in the linearized frame we must multiply by the width 
        //printf("\n sd_block X : = %d \n", sd_block_x) ;
        int sd_block_y = ( blockId_y + ( ( blockId_x + 1 )/ gridDim.x ) ) % gridDim.y ;
        //printf("\n sd block Y : = %d \n ", sd_block_y);
        int sd_coordinate = ( ( sd_block_y * width * subframe_height )    +
                            ( sd_block_x * subframe_height )    + 
                            ( gridDim.x * subframe_height * ( subframe_height - 1) + subframe_height - 1) ) * NUM_CHANNELS ;       
        //printf("\n sd coordinate: %d\n ", sd_coordinate );
        //printf("\nblock (%d, %d) will pick its sd from this location: %d, which is %d\n", blockId_x, blockId_y, sd_coordinate, input[sd_coordinate]);
        
        sd = input[ sd_coordinate ] ;

        d_sd_array[ (blockId_y * gridDim.x) + blockId_x ] = sd ;    

        diffusionSeq_fast_u4(startRow, endRow , startCol, endCol, width, sd, tile1, tile2, tile_byteStream, performInverseDiffusion ); //this is the sequential version based oon chebishev on the paper.. unfutunatly it is nonParallelizable

    //printf("\nsd of block(%d, %d)= %d\n", blockId_x, blockId_y, d_sd_array[(blockId_y * gridDim.x + blockId_x)]) ;
    
    } 
        
    __syncthreads();

    //reinterpret_cast<uchar3*>(output)[gy * width + gx] =
    //tile2[threadIdx.y * subframe_height + threadIdx.x] ;

    uchar4 out4 = tile2[threadIdx.y * subframe_height + threadIdx.x];
    reinterpret_cast<uchar3*>(output)[gy * width + gx] =
    make_uchar3(out4.x, out4.y, out4.z);

    //}
    //__syncthreads();
}

void diffusionOpWrapper( unsigned char *input, unsigned char *output, unsigned char *byteStreamFinal, int width, int height, int subframe_height, int performInverseDiffusion, std::vector<unsigned char>& sd_array) {

    const int total_pixels = width * height ;

    subframe_height = 12 ;

    const int num_subframesXdir = height / subframe_height ;
    const int num_subframesYdir = height / subframe_height ;

    int threadsXdir = subframe_height ;
    int threadsYdir = subframe_height ;
    
    printf("num_subframesXdir : %d\n", num_subframesXdir) ;
    printf("num_subframesYdir : %d\n", num_subframesYdir) ;

    unsigned char *d_input ;
    unsigned char *d_output ;
    unsigned char *d_byteStream ; // space which contains all of the byte streams that have to be applied to all of the subframes...
    unsigned char *d_sd_array ;

    //printf("total_pixels: %d\n", total_pixels) ;
    //printf("num_subframes: %d\n", num_subframes) ;

    cudaMalloc(&d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc(&d_output, total_pixels * sizeof( uchar3 ) ) ;
    cudaMalloc(&d_byteStream, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc( (void **) &d_sd_array, num_subframesXdir * num_subframesYdir * sizeof( unsigned char) ) ;
    cudaMemcpy(d_input, input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_byteStream, byteStreamFinal, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyHostToDevice) ;

    if( performInverseDiffusion == 1 ) {
        printf("sd values used for inverse diffusion\n");
        for( int i = 0; i < sd_array.size(); i ++ ){
            printf("%d", sd_array[i]); 
        }
        cudaMemcpy( d_sd_array, sd_array.data(), num_subframesXdir * num_subframesYdir * sizeof( unsigned char ), cudaMemcpyHostToDevice ) ;
    }

    //printf(".......");

    //for now exploiting the parallelism just at the block level ... and each block will run the algorithm serially...
    //will have to modify when optimizing.... either try to find another algorithm for diffusion different from the paper online which is parallelizable....
    //..or stick to this and just try to obtain the correct configurations of blocks which do make obtain the best performance... can see that while profiling i believe

    dim3 blocksPerGrid( num_subframesXdir , num_subframesYdir ) ;
    //dim3 threadsPerBlock( 1, 1 ) ; // maximum number of threads per block that can be allocated. // same way as confusion
    dim3 threadsPerBlock( subframe_height, subframe_height ) ;

    size_t sharedMemPerBlock =  3 * subframe_height * subframe_height * sizeof( uchar4 ) ;

    if(performInverseDiffusion == 0){
        diffusionKernel<<< blocksPerGrid, threadsPerBlock, sharedMemPerBlock >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, num_subframesXdir* num_subframesYdir, performInverseDiffusion, d_sd_array );
        printf("yolo1\n") ;
    } else {
        //inverseDiffusionKernel <<< blocksPerGrid, threadsPerBlock >>> ( d_input, d_output, d_byteStream, width, height, subframe_height, num_subframesXdir* num_subframesYdir, performInverseDiffusion, d_sd_array );
    }
    
    cudaDeviceSynchronize() ;
    cudaMemcpy(output, d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char), cudaMemcpyDeviceToHost) ;
    
    if( performInverseDiffusion == 0) {
        sd_array.resize( num_subframesXdir * num_subframesYdir ) ;
        cudaMemcpy( sd_array.data(), d_sd_array,  num_subframesXdir * num_subframesYdir * sizeof(unsigned char), cudaMemcpyDeviceToHost ) ;
        for(int i = 0; i < sd_array.size(); i ++){
            printf("%d ", sd_array[i]);
        }
    }

    cudaFree(d_input) ;
    cudaFree(d_byteStream) ;
    cudaFree(d_output) ;
    cudaFree(d_sd_array) ;
}