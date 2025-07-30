# Makefile for CUDA + OpenCV project

# Compiler and flags
NVCC := nvcc
CFLAGS := -O2
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)

# Files and output
TARGET := encryptApp
SRC := main.cpp encrypt_kernel.cu
OBJS := $(SRC:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS) -lcudart

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
