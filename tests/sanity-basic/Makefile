all: bin/write bin/read

CUDA_DIR ?= /usr/local/cuda
NVCC = $(CUDA_DIR)/bin/nvcc
INCLUDES = -I../../library/include
LIBS = -L../../library/lib64 -ldragon

bin/write: write.cu
	mkdir -p bin
	$(NVCC) $(INCLUDES) -o bin/write write.cu $(LIBS)

bin/read: read.cu
	mkdir -p bin
	$(NVCC) $(INCLUDES) -o bin/read read.cu $(LIBS)

clean:
	rm -rf bin

