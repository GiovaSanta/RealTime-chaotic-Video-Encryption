#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../../include/encrypt_kernel.hpp"
#include "../../include/prbg_main_plcm.hpp"
#include "../../include/confusion_kernel.hpp"
#include "../../include/diffusion_kernel.hpp"
#include <math.h>
#include <chrono>
#include "../../include/utils.hpp"

#define ROUNDS 5
#define NUM_CHANNELS 3 // number RGB channels

int main () {

//--------------------------------------------LOADING INPUT FRAME -------------------------------------------------------------------------

    unsigned char * h_frame_pinned = nullptr ;
    cudaMallocHost(&h_frame_pinned, 960 * 960 * NUM_CHANNELS * sizeof( unsigned char )) ;

    //cv::Mat inputFrame2 = cv::imread("testFrames/link960x960.png", cv::IMREAD_COLOR);
    //cv::Mat inputFrame = cv::imread("testFrames/ultraSmall4by4.png", cv::IMREAD_COLOR);
    //cv:: Mat inputFrame = cv::imread("testFrames/6by6test.png", cv::IMREAD_COLOR);

    cv::Mat inputFrame( 960, 960, CV_8UC3, h_frame_pinned ) ;
    cv::Mat tmp = cv::imread("testFrames/link960x960.png") ;
    tmp.copyTo(inputFrame) ;

    //cudaEvent_t evStart, evStop;
    //cudaEventCreate(&evStart);
    //cudaEventCreate(&evStop);

    //cudaStream_t stream;
    //cudaStreamCreate(&stream);

    //unsigned char *d_inputTry = nullptr;
    //cudaMalloc( (void **) &d_inputTry, NUM_CHANNELS * 960 * 960 * sizeof( unsigned char ) ) ;

    //cudaFree(0); //for initialization purposes
    //cudaMemcpyAsync(d_inputTry, inputFrame.data, NUM_CHANNELS * 960 * 960 * sizeof( unsigned char ), cudaMemcpyHostToDevice, stream);
    //cudaStreamSynchronize(stream);
    
    //cudaEventRecord(evStart, stream);
    //cudaMemcpyAsync(d_inputTry, inputFrame.data, NUM_CHANNELS * 960 * 960 * sizeof( unsigned char ) , cudaMemcpyHostToDevice, stream);   
    //cudaEventRecord(evStop, stream);
    //cudaEventSynchronize(evStop);

    //ms= 0.0f ;
    //cudaEventElapsedTime(&ms, evStart, evStop);
    //std::cout << "H2D copy time using pinned: " << (ms) << " ms\n"; 

    //quick check if frame exists
    
    if( inputFrame.empty() ) {
        std::cerr << "Failed to load Frame!\n" ;
        return -1 ;
    }
    
    int width = inputFrame.cols ;
    int height = inputFrame.rows ;

    int subframeWidth = 12 ;
    int subframeHeight = 8 ;
    //int subframeWidth = 3 ;
    //int subframeHeight = 2 ;

    //print_frameEncoding(inputFrame) ; // for debug purposes 
    int numSubFrames = ( width * height ) / ( subframeHeight * subframeWidth ) ;

    //printf ("width : %d, height: %d \n", width, height) ;
    //printf (" number of subframes: %d \n", numSubFrames ) ;

    double globalKey = 0.223456 ; //used for PRBGmain initialization. is a double value (0,0.5)
    double p = 0.3 ; //control parameter for the PRBGmain . is a value in (0,0.5)
    int numKeys = numSubFrames ;           
    int PRBGAiterations =  ( int )ceilf( ( ( float )( subframeHeight * subframeWidth * NUM_CHANNELS ) / 6.0 ) ) / 16 ; //( ( subframeWidth * subframeHeight ) / 6 ) + 1 ; 
    //const int PRBGAiterations =  ( int )ceilf( ( ( float )( subframeHeight * subframeWidth ) / 6.0 ) ) ;
    //PRBGAiterations = 1 * NUM_CHANNELS ;
    uint64_t sc ;

    int PRBGAsAmount = (width * height) / ( 6 ) ; // one prbga iteration produces 6 bytes.
    
    double *d_keysAndControlPs ; //array containing the control parameters and keys going to all the subframes.

    //printf("PRBGAs amount: %d\n", PRBGAsAmount) ;

    printf("PRBGA iterations: %d\n", PRBGAiterations); 

//--------------------------------------------CREATING PRBGMAIN KEYS AND MEASURING ITS EXECUTION TIME------------------------------------

    //auto start = std::chrono::high_resolution_clock::now() ; //measuiring execution time of generatePRBGmain keys function
    
    double* keysAndControlPsPinned = nullptr;
    cudaMallocHost(&keysAndControlPsPinned, ( 2 * PRBGAsAmount + 1 ) * sizeof( double ) ); //pinned memory
    generatePRBGMainKeysv3Pinned(globalKey, p, 2 * PRBGAsAmount + 1, &sc, keysAndControlPsPinned );

    //printf(" %f, %f\n", keysAndControlPsPinned[0], keysAndControlPsPinned[1]  ) ; //debug purpose

    //std::vector<double> keysAndControlPs = generatePRBGMainKeysv2( globalKey, p, 2*numKeys*16 + 1 , &sc ) ; // +1 for sc generation
    //std::vector<double> keysAndControlPs = generatePRBGMainKeysv2( globalKey, p, 2*PRBGAsAmount + 1 , &sc ) ; // +1 for sc generation

    //printMainKeys( keysAndControlPsPinned, 2*PRBGAsAmount ); //for debug purposes. prints controlPs and Keys later used in PRBGAs

    /*auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start ;
    std::cout << "Time taken for PRBG main: " << elapsed.count() << " ms\n" ; */ // this included to understand how long the main prbg takes

//---------------------------------------------DEFINING POINTER FOR inDEVICE ARRAYS -------------------------------------------------------

    std::vector<unsigned char> byteStreamFinal ; //byteStreamFinal array exists on the host only if you want to print for debug purposes.. else it lives inside the device 
    
    unsigned char *d_values4ByteStream_1 ; // first byte array containing all the byte streams of all the subframes.
    unsigned char *d_values4ByteStream_2 ; // second byte array containing all the byte streams of all the subframes.
    unsigned char *d_byteStreamFinal ; //final byte array ( which is a "xored" version of the first and second byte arrats ) containing all the byte arrays of all the subframes.
    
    unsigned char *d_sd_array ; 
    unsigned char *d_input ; //  array containing the encoding of all the frame itself before the confusion/diffusion
    unsigned char *d_output ; // array containing the encoding of all th13.325ms e frame itself after  the confusion/diffusion
    cudaFree(0); // for device initialization 

//--------------------------------------------MEMORY ALLOCATION FOR inDEVICE ARRAYS -------------------------------------------------------

    PRBGA_init( 2 * PRBGAsAmount,
                &d_keysAndControlPs, 
                &d_values4ByteStream_1, 
                &d_values4ByteStream_2, 
                &d_byteStreamFinal, 
                PRBGAiterations ) ;

    device_frame_allocation_old ( width * height, 
                                  &d_input, 
                                  &d_output,
                                  &d_sd_array, 
                                  numSubFrames ) ;

    PRBGAandByteStreamGenWrapper( d_keysAndControlPs, 
                                  //keysAndControlPs,
                                  keysAndControlPsPinned,
                                  (2 * PRBGAsAmount),
                                  d_values4ByteStream_1, 
                                  d_values4ByteStream_2, 
                                  d_byteStreamFinal, 
                                  byteStreamFinal,  
                                  PRBGAiterations, 
                                  subframeHeight, 
                                  subframeWidth, 
                                  width, 
                                  height ) ; 
    
    //print_ByteStreamFinal( byteStreamFinal, subframeHeight, subframeWidth, PRBGAiterations ); //debug purposes (works if cudamcpy DtH is uncommented activated in prbga wrapper)

    //cv::Mat current = inputFrame.clone() ;
    
    std::vector <unsigned char> buffer( width * height * NUM_CHANNELS ) ; // considering the rgb image , the image information is thrice that of the greyscale
    int performInverseConfusion = 0 ; 
    int performInverseDiffusion = 0 ; 

//-------------------------------------ENCRYPTION ROUNDS DIFFUSION AND CONFUSION--------------------------------------------------------------------

    cudaMemcpyAsync(d_input, inputFrame.data, width * height * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyHostToDevice) ;

    for( int i = 0; i < ROUNDS; i ++) {
    
        //CONFUSION
        confusionOpWrapper(inputFrame.data, d_input, inputFrame.data, d_output, width, height, subframeHeight, sc, performInverseConfusion ); 
        d_input = d_output ;
        
        //DIFFUSION
        diffusionOpWrapper( d_byteStreamFinal, d_input , inputFrame.data, d_output, inputFrame.data, width, height, subframeHeight, subframeWidth, performInverseDiffusion, d_sd_array ) ;
        d_input = d_output ;
    
    } 

    cudaMemcpy(inputFrame.data, d_output, width * height  * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyDeviceToHost); 
    
    //inputFrame = cv::Mat(height, width, CV_8UC3, buffer.data()).clone() ;
    std::string filenameConfusion = "testResults/afterEncryption.png" ;  //final encoded frame after ALL rounds of confusion and diffusion.
    cv::imwrite(filenameConfusion, inputFrame ); 

    performInverseDiffusion = 1 ;
    performInverseConfusion = 1 ; // doing the inverse of the confusiion to see if we are able to obtain the original image from the confused one.

    for( int i = ROUNDS-1 ; i >=0; i --) {

        //INVERSE OF DIFFUSION
        diffusionOpWrapper( d_byteStreamFinal, d_input, inputFrame.data, d_output, inputFrame.data, width, height, subframeHeight, subframeWidth, performInverseDiffusion, d_sd_array ) ;
        d_input = d_output ;

        //INVERSE OF CONFUSION 
        confusionOpWrapper(inputFrame.data, d_input, inputFrame.data, d_output, width, height, subframeHeight, sc, performInverseConfusion );
        d_input = d_output ;
    } 

    cudaMemcpy(inputFrame.data, d_output, width * height  * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyDeviceToHost);
    
    inputFrame = cv::Mat( height, width, CV_8UC3, inputFrame.data ).clone() ;
    std::string filenameInvConfusion = "testResults/afterInvEncription.png" ;
    cv::imwrite(filenameInvConfusion, inputFrame ); 

    freeMemory( d_byteStreamFinal, d_input, d_output ) ;

    cudaFreeHost( h_frame_pinned ) ;
    cudaFreeHost( keysAndControlPsPinned ); 
    //cudaFree(d_inputTry);

    return 0;
}

void PRBGA_init( int numParameters,
                 double ** d_keysAndControlPs,
                 unsigned char ** d_values4ByteStream_1, 
                 unsigned char ** d_values4ByteStream_2, 
                 unsigned char ** d_byteStreamFinal, 
                 int PRBGAiterations  ) {

    printf("numParameters: %d\n", numParameters);
    int numKeys = numParameters/2 ;

    int totalSize = numKeys * PRBGAiterations * 6 ; 
    //int totalSize = numKeys * PRBGAiterations * 6 ;

    printf("totalSize: %d\n", totalSize );

    cudaMalloc( ( void ** ) d_keysAndControlPs, numParameters * sizeof( double ) ) ;
    cudaMalloc( ( void ** ) d_values4ByteStream_1, totalSize * sizeof( unsigned char ) ) ;
    cudaMalloc( ( void ** ) d_values4ByteStream_2, totalSize * sizeof( unsigned char ) ) ;  
    cudaMalloc( ( void ** ) d_byteStreamFinal, totalSize * sizeof( unsigned char ) ) ;

    return;
}

//function used to allocate space for the arrays containing frame encoding used during the confusion and diffusion
void device_frame_allocation_old ( int total_pixels, unsigned char ** d_input, unsigned char ** d_output, unsigned char ** d_sd_array, int numSubFrames  ) {

    cudaMalloc( (void **) d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc( (void **) d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    cudaMalloc( (void **) d_sd_array, numSubFrames * sizeof( unsigned char ) ) ; //used for sd seeds allocation for diffusion

    //printf("total_pixels: %d", total_pixels) ;

    return ;
}

void print_ByteStreamFinal( std::vector<unsigned char> byteStreamFinal, int subframeHeight, int subframeWidth, int PRBGAiterations) {

    int j = 0 ;

    printf( "\nbyteStreamFinal size: %d\n", byteStreamFinal.size() ) ;
    for ( int i = 0; i < byteStreamFinal.size(); i++ ) {
        if( (i % (subframeHeight * subframeWidth * PRBGAiterations ) ) == 0 ) {
            printf("\n%d\n", j) ;
            j++;
        }
        printf("%d ", byteStreamFinal[i]) ;
    }
    printf("\n"); 

    return ;

}

void print_frameEncoding( const cv::Mat& img ) {

    printf("\nprinting all the pixel encodings of the frame: \n");

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(x, y);  // B, G, R
            std::cout << "Pixel (" << x << "," << y << "): "
                      << "B=" << (int)pixel[0] << " "
                      << "G=" << (int)pixel[1] << " "
                      << "R=" << (int)pixel[2] << std::endl;    
        } 
    } 

    return;
}

void freeMemory( unsigned char *d_byteStreamFinal, unsigned char * d_input, unsigned char * d_output ) {

    cudaFree( d_byteStreamFinal );
    cudaFree( d_input ) ;
    cudaFree( d_output ) ;

    return ; 
}

//prints the array containing control parameters and keys for later prbgas
void printMainKeys ( double * keysAndControlPs, int N ) {

    for( int i = 0; i< N; i++){
        if(i % 2 == 0 ) { printf("\n"); }
    
        printf( " %f ", keysAndControlPs[i] ) ;
    } 
    printf("\n") ;
    
}