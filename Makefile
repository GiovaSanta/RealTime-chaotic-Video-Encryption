# Makefile for CUDA + OpenCV project (with optional NVTX for Nsight Systems)

NVCC := nvcc

# ---- Base flags ----
CFLAGS := -O2 -lineinfo
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)

# ---- Optional NVTX ----
# Enable with: make NVTX=1
NVTX ?= 0
ifeq ($(NVTX),1)
  CFLAGS += -DUSE_NVTX
  NVTX_LIBS := -lnvToolsExt
else
  NVTX_LIBS :=
endif

# ---- Target / sources ----
TARGET := encryptApp
SRC := src/main.cpp src/prbg_main_plcm.cpp src/prbga_kernel_Final.cu src/confusion_kernel_Final.cu src/diffusion_kernel_Final.cu

# ---- Build ----
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS) -lcudart $(NVTX_LIBS) -Xcompiler -pthread

clean:
	rm -f $(TARGET) *.o src/*.o

.PHONY: all clean
