# Makefile for CUDA + OpenCV project

# Compiler and flags
NVCC := nvcc
CFLAGS := -O2
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)

# Files and output
TARGET := encryptApp
#SRC := src/main.cpp src/prbg_main_plcm.cpp src/prbga_kernel.cu src/confusion_kernel.cu src/diffusion_kernel.cu

SRC := src/main.cpp src/prbg_main_plcm.cpp src/prbga_kernel.cu src/confusion_kernelVersion2.cu src/diffusion_kernelv4.cu

OBJS := $(SRC:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS) -lcudart

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
