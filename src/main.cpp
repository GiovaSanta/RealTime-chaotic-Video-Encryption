#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdint.h>
#include "encrypt_kernel.hpp"

int main () {

    //load the image (grayScale mode for now)
    cv::Mat inputImg = cv::imread("initialTestFrame.png", cv::IMREAD_GRAYSCALE);
    if( inputImg.empty()) {
        std::cerr << "Failed to load Image!\n";
        return -1;
    }
    int width = inputImg.cols;
    int height = inputImg.rows;
   
    std::vector<uint8_t> encrypted_data(width * height);

    //calling the wrapper
    encryptFrame(inputImg.data, encrypted_data.data(), width, height);

    //Copying the result back to the host environment
    cv::Mat outputImg(height, width, CV_8UC1, encrypted_data.data());
    
    cv::imwrite("encrypted_output.png", outputImg);

    std::cout << "Encryption complete. Saved as encrypted_output.png\n";
    
    return 0;

}