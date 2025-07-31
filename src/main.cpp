#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdint.h>
#include "../include/encrypt_kernel.hpp"
#include "../include/prbg_main_plcm.hpp"

int main () {

    double globalKey = 0.123456 ; //used for PRBGmain initialization. is a double value (0,0.5)
    double p = 0.3; //control parameter for the PRBGmain . is a value in (0,0.5)
    int numKeys =128 ;  // each iteration of the PRBG main is a resultant input seed...
                        //...for the future subsequent PRBGas as described by article
    

    std::vector<double> keys = generatePRBGMainKeys( globalKey, p, 128 );
    //load the Frame (grayScale mode for now)
    cv::Mat inputFrame = cv::imread("../testFrames/initialTestFrame.png", cv::IMREAD_GRAYSCALE);
    
    //quick check if frame exists
    if( inputFrame.empty()) {
        std::cerr << "Failed to load Frame!\n";
        return -1;
    }
   
    cv::Mat encryptedFrame(inputFrame.size(), inputFrame.type() );

    float seed = 0.7f; // for PRBG
    float k = 0.5f; // control parameter for PiecewiseLinearChaoticMap (PLCM)


    //calling the wrapper for the kernel
    //encryptFrame(inputFrame, encryptedFrame, seed, k);

    //Copying the result back to the host environment
    cv::imwrite("../testFrames/encrypted_output.png", encryptedFrame);

    std::cout << "Encryption complete. Saved as encrypted_output.png\n";
    
    return 0;

}