#SUBDIRS = programs/lib programs/cudamemcpy programs/uvm programs/hostreg programs/nvmgpu

CLEANFILES = *~
DISTCLEANFILES = *~

SRCS = backprop_train.cu backprop.c backprop_cuda_kernel.cuh backprop.h bpnn.c bpnn.h cudaio.c timer.c timer.h

noinst_PROGRAMS = backprop.nvmgpu backprop.hostreg backprop.uvm backprop.cudamemcpy generator

backprop_nvmgpu_SOURCES = $(SRCS)
backprop_hostreg_SOURCES = $(SRCS)
backprop_uvm_SOURCES = $(SRCS)
backprop_cudamemcpy_SOURCES = $(SRCS)
generator_SOURCES = $(SRCS) generator.c

backprop_nvmgpu_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -DNVMGPU
backprop_nvmgpu_LDADD = -lcuda -lcudart $(top_srcdir)/library/src/libdragon.a $(GLIB_LIBS)
backprop_nvmgpu_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_hostreg_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -DHOSTREG
backprop_hostreg_LDADD = -lcuda -lcudart
backprop_hostreg_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_uvm_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -DUVM
backprop_uvm_LDADD = -lcuda -lcudart
backprop_uvm_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_cudamemcpy_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -DCUDAMEMCPY
backprop_cudamemcpy_LDADD = -lcuda -lcudart
backprop_cudamemcpy_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

generator_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -DGENERATOR
generator_LDADD = -lcuda -lcudart
generator_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
