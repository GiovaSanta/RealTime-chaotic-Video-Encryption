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

#define ROUNDS 1
#define NUM_CHANNELS 3 // number RGB channels

int main () {

    //int device = 0;  // use device 0 (default)
    //cudaDeviceProp prop;
    //cudaError_t status = cudaGetDeviceProperties(&prop, device);

    //printf("GPU Name: %s\n", prop.name);
    //printf("Number of Streaming Multiprocessors: %d\n", prop.multiProcessorCount);

    //std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    //std::cout << "Max grid dimensions: (" << maxGridDimX << ", "<< maxGridDimY << ", " << maxGridDimZ << ")" << std::endl;

    //load the Frame (grayScale mode for now)
    cv::Mat inputFrame = cv::imread("testFrames/link960x960.png", cv::IMREAD_COLOR);
    //cv::Mat inputFrame = cv::imread("testFrames/shrekAndDonkey768x768.png", cv::IMREAD_COLOR);
    //cv::Mat inputFrame = cv::imread("testFrames/ultraSmall4by4.png", cv::IMREAD_COLOR);
    //cv::Mat inputFrame = cv::imread("testFrames/weird8by8.png", cv::IMREAD_COLOR);
    //cv::Mat inputFrame = cv::imread("testFrames/weird8by8.png", cv::IMREAD_COLOR);

    //quick check if frame exists
    if( inputFrame.empty() ) {
        std::cerr << "Failed to load Frame!\n" ;
        return -1 ;
    }
    
    int width = inputFrame.cols ;
    int height = inputFrame.rows ;
    //int subFrameHeight = 6 ; // keeping the height of the subframe as fixed parameter for now to compute the num of subframes to allocate

    int subframeWidth = 12 ;
    int subframeHeight = 8 ;

    //printing all the pixel encodings of the frame.

    /*printf("\nprinting all the pixel encodings of the frame: \n");
    for (int y = 0; y < inputFrame.rows; y++) {
        for (int x = 0; x < inputFrame.cols; x++) {
            cv::Vec3b pixel = inputFrame.at<cv::Vec3b>(x, y);  // B, G, R
            std::cout << "Pixel (" << x << "," << y << "): "
                      << "B=" << (int)pixel[0] << " "
                      << "G=" << (int)pixel[1] << " "
                      << "R=" << (int)pixel[2] << std::endl;    
        } 
    } */


    //printing the encoding values that should be related to the sd that should be chosen for the subframes during diffusion:

    /*printf("\npixel values related to the sd values that need to be chosen in diffusion: \n");
    for (int y = 0; y < inputFrame.rows; y++) {
        for (int x = 0; x < inputFrame.cols; x++) {
            if( ( (x+1) % subframeWidth == 0 ) && ( (y + 1 ) % subframeHeight == 0 ) ) {
                cv::Vec3b pixel = inputFrame.at<cv::Vec3b>(x, y);  // B, G, R
                std::cout << "Pixel (" << x << "," << y << "): "
                          << "B=" << (int)pixel[0] << " "
                          << "G=" << (int)pixel[1] << " "
                          << "R=" << (int)pixel[2] << std::endl;    
            }
        }
    }*/
    
    int numSubFrames = ( width * height ) / ( subframeHeight * subframeWidth ) ;

    //printf ("width : %d, height: %d \n", width, height) ;
    //printf (" number of subframes: %d \n", numSubFrames ) ;

    double globalKey = 0.223456 ; //used for PRBGmain initialization. is a double value (0,0.5)
    double p = 0.3 ; //control parameter for the PRBGmain . is a value in (0,0.5)

    int numKeys = numSubFrames;  /* each iteration of the PRBG main is a resultant input seed...
                        ...for the future subsequent PRBGas as described by article
                        considering a 768 x 768 Frame.
                        considering that 128 blocks (equivalent to the assistant threads in the paper) will manage 128 subframes
                        then we have 128 6x768 subframes.
                        a particular diffusion operation on a specifc subframe involves 6*768 pixels.
                        in particular, each of the 6* 768 pixels has a corresponding byte associated for diffusion operation.
                        so for a subframe, 768* 6 bytes are asssociated to it.
                        the article says that one result of a particular PRBGa will form 6 bytes of the bytestream used in diffusion in a particular subframe..
                        hence, the amount of outputs of the PRBGA is 768 * 6 / 6 = 768.
                        the byte stream of a specific subframe is that of dimension 768*6 .
                        */
                       
    const int PRBGAiterations =  ( int )ceilf( ( ( float )( subframeHeight * subframeWidth  ) / 6.0 ) ) / 16 ; //( ( subframeWidth * subframeHeight ) / 6 ) + 1 ; 
    //const int PRBGAiterations =  ( int )ceilf( ( ( float )( subframeHeight * subframeWidth ) / 6.0 ) ) ;

    //printf( "number of total keys for all the PRBGa components: %d\n", numKeys ) ;
    //printf( "prbga iterations for the singular prbga: %d\n", PRBGAiterations ) ;
    uint64_t sc ;

    auto start = std::chrono::high_resolution_clock::now() ;

    //generating the keys and control parameters that will be fed to future PRBGas
    std::vector<double> keysAndControlPs = generatePRBGMainKeysv2( globalKey, p, 2*numKeys*16 + 1 , &sc ) ; // +1 for sc generation
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start ;
    std::cout << "Time taken for PRBG main: " << elapsed.count() << " ms\n" ; // this included to understand how long the main prbg takes

    //printf("the confusion seed: %d\n", sc);
    std::vector<unsigned char> byteStreamFinal, output1, output2, sd_array ;
    double *d_keysAndControlPs ;
    unsigned char *d_values4ByteStream_1 ;
    unsigned char *d_values4ByteStream_2 ;
    unsigned char *d_byteStreamFinal ;
    
    //now that the parameter needed to initialize the various PRBGAs are ready, we can launch the wrapper to the kernel dealing with beginning PRBGAs execution
    
    cudaFree(0);
    PRBGAinit( keysAndControlPs.size(), &d_keysAndControlPs, &d_values4ByteStream_1, &d_values4ByteStream_2, &d_byteStreamFinal, PRBGAiterations ) ;
    
    PRBGAandByteStreamGenWrapper( d_keysAndControlPs, keysAndControlPs, d_values4ByteStream_1, d_values4ByteStream_2, d_byteStreamFinal, byteStreamFinal, output1, output2, PRBGAiterations, subframeHeight, subframeWidth, width, height ) ;

    //print_ByteStreamFinal( byteStreamFinal, subframeHeight, subframeWidth); //debug purposes

    // To access result for block 0 (contains the values generated by the PRBGA assigned to block 0), for example:

    cv::Mat current = inputFrame.clone() ;
    std::vector <unsigned char> buffer( width * height * NUM_CHANNELS ) ; // considering the rgb image , the image information is thrice that of the greyscale
    
    int performInverseConfusion = 0 ; 
    int performInverseDiffusion = 0 ; 
    
    //std::vector <unsigned char> buffer1( width * height * 4 ) ;
    //cv::Mat imgRGBA;
    //cv::cvtColor(inputFrame, imgRGBA, cv::COLOR_BGR2RGBA);
    //runCoalescedLoadWrapper(imgRGBA.data, buffer1.data(), width, height );
    //runCoalescedLoadWrapper( inputFrame.data, buffer.data(), width, height ) ;

    for( int i = 0; i< ROUNDS; i ++){
    
        //CONFUSION
        /*confusionOpWrapper(current.data, buffer.data(), width, height, subframeHeight, sc, performInverseConfusion); 
        current = cv::Mat(height, width, CV_8UC3, buffer.data()).clone();
        std::string filenameConfusion = "testResults/afterConfusion" + std::to_string(i) +".png" ;
        cv::imwrite(filenameConfusion, current ); */
    
        //DIFFUSION

        /*diffusionOpWrapper( current.data, buffer.data(), byteStreamFinal.data(), width, height, subframeHeight, subframeWidth, performInverseDiffusion, sd_array ) ;
        current = cv::Mat(height, width, CV_8UC3, buffer.data()).clone();
        std::string filenameDiffusion = "testResults/afterDiffusion" + std::to_string(i) +".png" ;
        cv::imwrite(filenameDiffusion, current ) ;  */

        /*
        printf("\nsd array size: %d\n", sd_array.size() ) ;
        printf("\nsd array: \n") ; 
        for( int i = 0; i < sd_array.size(); i ++) {
            printf("sd[%d] = %d\n", i, sd_array[i] ) ;  
        }
        */
    }

    performInverseDiffusion = 1;
    performInverseConfusion = 1; // doing the inverse of the confusiion to see if we are able to obtain the original image from the confused one.

    for( int i = ROUNDS-1 ; i >=0; i --) {

        //INVERSE OF DIFFUSION
        //here performing the inverse of the diffusion to see if i am able to obtain the original frame back implying diffusion operations make sense.
        
        /*diffusionOpWrapper( current.data, buffer.data(), byteStreamFinal.data(), width, height, subframeHeight, subframeWidth, performInverseDiffusion, sd_array) ;
        current = cv::Mat(height, width, CV_8UC3, buffer.data()).clone() ;
        std::string filenameInvDiffusion = "testResults/afterInvDiffusion" + std::to_string(i) + ".png" ;
        cv::imwrite( filenameInvDiffusion, current) ;    */

        //INVERSE OF CONFUSION 
        /*confusionOpWrapper(current.data, buffer.data(), width, height, subframeHeight, sc, performInverseConfusion );
        current = cv::Mat(height, width, CV_8UC3, buffer.data()).clone() ;
        std::string filenameInvConfusion = "testResults/afterInvConfusion" + std::to_string(i) + ".png" ;
        cv::imwrite(filenameInvConfusion, current ); */
    
    }

    return 0;
}

void PRBGAinit( int numParameters, double ** d_keysAndControlPs, unsigned char ** d_values4ByteStream_1, unsigned char **d_values4ByteStream_2, unsigned char ** d_byteStreamFinal, int PRBGAiterations  ){

    int numKeys = numParameters/2 ;

    int totalSize = numKeys * PRBGAiterations * 6 ; 

    cudaMalloc( (void **) d_keysAndControlPs, numParameters * sizeof( double ) ) ;
    cudaMalloc( (void **) d_values4ByteStream_1, totalSize *  sizeof( unsigned char ) ) ;
    cudaMalloc( (void **) d_values4ByteStream_2, totalSize * sizeof( unsigned char ) ) ; 
    cudaMalloc( (void **) d_byteStreamFinal, totalSize * sizeof( unsigned char ) ) ; 
    
    return;

}

void print_ByteStreamFinal( std::vector<unsigned char> byteStreamFinal, int subframeHeight, int subframeWidth ) {

    int j = 0 ;

    printf( "\nbyteStreamFinal size: %d\n", byteStreamFinal.size() ) ;
    for ( int i = 0; i < byteStreamFinal.size(); i++ ) {
        if( (i % (subframeHeight * subframeWidth ) ) == 0 ) {
            printf("\n%d\n", j) ;
            j++;
        }
        printf("%d ", byteStreamFinal[i]) ;
    }
    printf("\n"); 

    return ;

}