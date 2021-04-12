all: bin/write bin/read

GLIB_LIBS ?= $(shell pkg-config --libs glib-2.0)
CUDA_DIR ?= /usr/local/cuda
NVCC = $(CUDA_DIR)/bin/nvcc
INCLUDES = -I../../library/include
LIBS = -L../../library/lib64 -ldragon $(GLIB_LIBS) -lcudart

bin/write: write.cu
	mkdir -p bin
	$(NVCC) $(INCLUDES) -o bin/write write.cu $(LIBS)

bin/read: read.cu
	mkdir -p bin
	$(NVCC) $(INCLUDES) -o bin/read read.cu $(LIBS)

clean:
	rm -rf bin

