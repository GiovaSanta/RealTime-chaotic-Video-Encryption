#ifndef UTILS_HPP
#define UTILS_HPP

void print_ByteStreamFinal( std::vector<unsigned char> byteStreamFinal, int subframeHeight, int subframeWidth, int PRBGAiterations );

void print_frameEncoding( const cv::Mat& img ) ;

void freeMemory( unsigned char *d_byteStreamFinal ) ;

void device_frame_allocation ( int total_pixels, unsigned char ** d_sd_array, int numSubFrames  ) ;

void printMainKeys ( double * keysAndControlPs, int N ) ;

static int computeBitrate(int w, int h, int fps) ;

#endif