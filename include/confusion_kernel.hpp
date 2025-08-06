#ifndef CONFUSION_KERNEL_HPP
#define CONFUSION_KERNEL_HPP

#include <vector>
#include <opencv2/opencv.hpp> 
#include <cstdint>

void confusionOpWrapper( unsigned char *input, unsigned char *output, int width , int height, int sc, int performInverseConfusion);

#endif 