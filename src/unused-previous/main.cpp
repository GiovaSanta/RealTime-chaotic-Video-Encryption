#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../include/encrypt_kernel.hpp"
#include "../include/prbg_main_plcm.hpp"
#include "../include/confusion_kernel.hpp"
#include "../include/diffusion_kernel.hpp"
#include <math.h>
#include <chrono>
#include "../include/utils.hpp"

#define ROUNDS 5
#define NUM_CHANNELS 3 // number RGB channels

int main () {

//---------------------------------------------DEFINING POINTER FOR inDEVICE ARRAYS -------------------------------------------------------

    std::vector<unsigned char> byteStreamFinal ; //byteStreamFinal array exists on the host only if you want to print for debug purposes.. else it lives inside the device 
    
    unsigned char *d_values4ByteStream_1 ; // first byte array containing all the byte streams of all the subframes.
    unsigned char *d_values4ByteStream_2 ; // second byte array containing all the byte streams of all the subframes.
    unsigned char *d_byteStreamFinal ; //final byte array ( which is a "xored" version of the first and second byte arrats ) containing all the byte arrays of all the subframes.
    
    unsigned char *d_sd_array ; 
    unsigned char *d_input[2] ; //  array containing the encoding of all the frame itself before the confusion/diffusion
    unsigned char *d_output[2] ; // array containing the encoding of all th13.325ms e frame itself after  the confusion/diffusion
    
    double *d_keysAndControlPs ; //array containing the control parameters and keys going to all the subframes.
    
    double* keysAndControlPsPinned = nullptr;
    unsigned char* h_frame_pinned[2] ;  //containing frame iamge in host 
    
    int width = 960 ; //width of frame
    int height = 960 ; //height of frame
    int PRBGAsAmount = (width * height) / ( 6 ) ; // one prbga iteration produces 6 bytes.
    
    int subframeWidth = 12 ;
    int subframeHeight = 8 ;
    int PRBGAiterations =  ( int )ceilf( ( ( float )( subframeHeight * subframeWidth * NUM_CHANNELS ) / 6.0 ) ) / 16 ; //( ( subframeWidth * subframeHeight ) / 6 ) + 1 ; 
    int numSubFrames = ( width * height ) / ( subframeHeight * subframeWidth ) ;

    int frameCount = 0 ;

    cv::Mat frame;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);  

    cudaEvent_t frameReady[2]; //for double buffering purposes
    cudaEventCreate(&frameReady[0]);
    cudaEventCreate(&frameReady[1]);

    for (int i = 0; i < 2; i++) {

        // Host pinned buffer (CPU side)
        cudaMallocHost(&h_frame_pinned[i], width * height * NUM_CHANNELS * sizeof(unsigned char) );

        // Device buffers (GPU side)
        cudaMalloc(&d_input[i],  width * height * NUM_CHANNELS * sizeof(unsigned char) );
        cudaMalloc(&d_output[i], width * height * NUM_CHANNELS * sizeof(unsigned char) );
    }

    //cudaMallocHost(&h_frame_pinned, width * height * NUM_CHANNELS * sizeof( unsigned char )) ;
    cudaMallocHost(&keysAndControlPsPinned, ( 2 * PRBGAsAmount + 1 ) * sizeof( double ) ); //pinned memory

    PRBGA_init( 2 * PRBGAsAmount, //allocation function
            &d_keysAndControlPs, 
            &d_values4ByteStream_1, 
            &d_values4ByteStream_2, 
            &d_byteStreamFinal, 
            PRBGAiterations ) ;

    device_frame_allocation ( width * height, 
                              &d_sd_array, 
                              numSubFrames ) ;

    cv::VideoCapture cap("testVideos/test_960x960_20fps.mp4") ;
    //cv::VideoCapture cap("testVideos/test_768x768_20fps.mp4") ;

    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file\n";
        return -1;
    }

    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Total frames = " << totalFrames << std::endl;

    cv::VideoWriter writer( "testResults/encrypted_output.mp4", cv::VideoWriter::fourcc('a','v','c','1'), 20, cv::Size(width, height), true );

    if (!writer.isOpened()) {
        std::cerr << "ERROR: Could not open VideoWriter\n";
        return -1;
    }

//--------------------------------------------LOADING INPUT FRAME -------------------------------------------------------------------------

    //print_frameEncoding(inputFrame) ; // for debug purposes 

    double globalKey = 0.223456 ; //used for PRBGmain initialization. is a double value (0,0.5)
    double p = 0.3 ; //control parameter for the PRBGmain . is a value in (0,0.5)
    int numKeys = numSubFrames ;           
    uint64_t sc ;

//--------------------------------------------CREATING PRBGMAIN KEYS AND MEASURING ITS EXECUTION TIME------------------------------------
    
    //printMainKeys( keysAndControlPsPinned, 2*PRBGAsAmount ); //for debug purposes. prints controlPs and Keys later used in PRBGAs

//--------------------------------------------MEMORY ALLOCATION FOR inDEVICE ARRAYS -------------------------------------------------------

    auto t0 = std::chrono::high_resolution_clock::now();

    //cap.read(frame)
    while (  frameCount < totalFrames ) {

    frame = cv::Mat(height, width, CV_8UC3, cv::Scalar(128,128,128));

    int cur  = frameCount % 2;          // GPU writes here
    int prev = (frameCount + 1) % 2;    // CPU reads here

    generatePRBGMainKeysv3Pinned(globalKey, p, 2 * PRBGAsAmount + 1, &sc, keysAndControlPsPinned ); //runs on CPU
    
    memcpy( h_frame_pinned[cur], frame.data, width * height * NUM_CHANNELS ); // copy from paged to pinned 
    
    PRBGAandByteStreamGenWrapper( d_keysAndControlPs, 
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
                                  height,
                                  stream1 ) ;

    //print_ByteStreamFinal( byteStreamFinal, subframeHeight, subframeWidth, PRBGAiterations ); //debug purposes (works if cudamcpy DtH is uncommented activated in prbga wrapper)
    //6.5 ms

    int performInverseConfusion = 0 ; 
    int performInverseDiffusion = 0 ; 

//-------------------------------------ENCRYPTION ROUNDS DIFFUSION AND CONFUSION--------------------------------------------------------------------

    cudaMemcpyAsync(d_input[cur], h_frame_pinned[cur], width * height * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyHostToDevice, stream1) ;

    for( int i = 0; i < ROUNDS; i ++) {
    
        //CONFUSION
        confusionOpWrapper(frame.data, d_input[cur], frame.data, d_output[cur], width, height, subframeHeight, sc, performInverseConfusion, stream1 ); 
        std::swap(d_input[cur], d_output[cur]);
        
        //DIFFUSION
        diffusionOpWrapper( d_byteStreamFinal, d_input[cur] , frame.data, d_output[cur], frame.data, width, height, subframeHeight, subframeWidth, performInverseDiffusion, d_sd_array, stream1 ) ;
        std::swap(d_input[cur], d_output[cur]) ;
    
    } 

    cudaMemcpyAsync(h_frame_pinned[cur], d_input[cur], width * height  * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyDeviceToHost, stream1);
    cudaEventRecord(frameReady[cur], stream1);    
    
    if (frameCount > 0) {
        cudaEventSynchronize(frameReady[prev]);   // wait ONLY for previous buffer
        //auto start = std::chrono::high_resolution_clock::now() ;
        cv::Mat encryptedFrame(height, width, CV_8UC3, h_frame_pinned[prev]);
        //writer.write(encryptedFrame);
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> elapsed = end - start ;
        //std::cout << "Time taken for PRBG main: " << elapsed.count() << " ms\n" ;
    }

    //cv::Mat encryptedFrame(height, width, CV_8UC3, h_frame_pinned) ;
    //writer.write(encryptedFrame) ;

    /*if( frameCount % 100 == 0 ) {
         
        //cudaStreamSynchronize(stream); 
        //cudaDeviceSynchronize();  
        std::string filenameConfusion = "testResults/afterEncryption" + std::to_string(frameCount) + ".png" ;  //final encoded frame after ALL rounds of confusion and diffusion.
        cv::Mat encryptedMat(height, width, CV_8UC3, h_frame_pinned);
        cv::imwrite(filenameConfusion, encryptedMat);   
    }*/

    performInverseDiffusion = 1 ;
    performInverseConfusion = 1 ; // doing the inverse of the confusiion to see if we are able to obtain the original image from the confused one.

    /*for( int i = ROUNDS-1 ; i >=0; i --) {

        //INVERSE OF DIFFUSION
        diffusionOpWrapper( d_byteStreamFinal, d_input, frame.data, d_output, frame.data, width, height, subframeHeight, subframeWidth, performInverseDiffusion, d_sd_array, stream ) ;
        std::swap(d_input, d_output) ;

        //INVERSE OF CONFUSION 
        confusionOpWrapper(frame.data, d_input, frame.data, d_output, width, height, subframeHeight, sc, performInverseConfusion, stream);
        std::swap(d_input, d_output) ;
    } */

    /*if( frameCount % 100 == 0) {
        cudaMemcpyAsync(h_frame_pinned, d_input, width * height * NUM_CHANNELS * sizeof( unsigned char ), cudaMemcpyDeviceToHost);
        //cudaStreamSynchronize(stream);   
        std::string filenameInvConfusion = "testResults/afterInvEncription" + std::to_string(frameCount) + ".png" ;
        cv::Mat decryptedMat(height, width, CV_8UC3, h_frame_pinned);
        cv::imwrite(filenameInvConfusion, decryptedMat);
    }*/
    
    /*
    cv::Mat decryptedFrame(height, width, CV_8UC3, h_frame_pinned);
    writer.write(decryptedFrame); */

    frameCount ++ ;    
    }

    //fps measuring
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds =
    std::chrono::duration<double>(t1 - t0).count();
    std::cout << "FPS: " << frameCount / seconds << std::endl;

    cudaStreamDestroy(stream1);

    cudaEventDestroy(frameReady[0]);
    cudaEventDestroy(frameReady[1]);
    
    freeMemory( d_byteStreamFinal ) ;
    for (int i = 0; i < 2; i++) {
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
    }
    cudaFreeHost( h_frame_pinned[0] ) ;
    cudaFreeHost( h_frame_pinned[1] ) ;
    cudaFreeHost( keysAndControlPsPinned ) ; 
    cudaFree(d_keysAndControlPs);
    cudaFree(d_values4ByteStream_1);
    cudaFree(d_values4ByteStream_2);

    //cudaFree(d_inputTry);

    return 0;
}

void PRBGA_init( int numParameters,
                 double ** d_keysAndControlPs,
                 unsigned char ** d_values4ByteStream_1, 
                 unsigned char ** d_values4ByteStream_2, 
                 unsigned char ** d_byteStreamFinal, 
                 int PRBGAiterations  ) {

    //printf("numParameters: %d\n", numParameters);
    int numKeys = numParameters/2 ;

    int totalSize = numKeys * PRBGAiterations * 6 ; 
    //int totalSize = numKeys * PRBGAiterations * 6 ;

    //printf("totalSize: %d\n", totalSize );

    cudaMalloc( ( void ** ) d_keysAndControlPs, numParameters * sizeof( double ) ) ;
    cudaMalloc( ( void ** ) d_values4ByteStream_1, totalSize * sizeof( unsigned char ) ) ;
    cudaMalloc( ( void ** ) d_values4ByteStream_2, totalSize * sizeof( unsigned char ) ) ;  
    cudaMalloc( ( void ** ) d_byteStreamFinal, totalSize * sizeof( unsigned char ) ) ;

    return;
}

//function used to allocate space for the arrays containing frame encoding used during the confusion and diffusion
void device_frame_allocation ( int total_pixels, unsigned char ** d_sd_array, int numSubFrames  ) {

    //cudaMalloc( (void **) d_input, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
    //cudaMalloc( (void **) d_output, total_pixels * NUM_CHANNELS * sizeof( unsigned char ) ) ;
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

void freeMemory( unsigned char *d_byteStreamFinal ) {

    cudaFree( d_byteStreamFinal );
    //cudaFree( d_input ) ;
    //cudaFree( d_output ) ;

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