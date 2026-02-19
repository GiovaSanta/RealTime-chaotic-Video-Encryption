# RealTime-chaotic-Video-Encryption
extended study of Real-time chaotic video encryption algorithms based on multithreaded parallel confusion and diffusion. An extended work of Dong Jiang article (found in this repository) adapting the approach to an NVIDIA Jetson Nano GPU

GENERAL INSTRUCTIONS. to be executed when connected to jetson nano

for choosing the video, consult the testVideos folder for available videos of different sizes that can be encrypted. 
go to line 139 of main.cpp to choose the video to encrypt.

To compile the program, the make command can be executed in the cli.

To remove a created executable, can use make clean.

For running the executable after compilation is finished , perform "./encryptApp"

The resulting full encrypted video can be found in the testResults folder along its decrypted counter part.