CLEANFILES = *~
DISTCLEANFILES = *~

SRCS_COMMON = backprop_train.cu backprop_cuda_kernel.cuh backprop.h bpnn.c bpnn.h cudaio.c
SRCS = backprop.c $(SRCS_COMMON)
noinst_PROGRAMS = backprop.nvmgpu backprop.hostreg backprop.uvm backprop.cudamemcpy generator

backprop_nvmgpu_SOURCES = $(SRCS)
backprop_hostreg_SOURCES = $(SRCS)
backprop_uvm_SOURCES = $(SRCS)
backprop_cudamemcpy_SOURCES = $(SRCS)
generator_SOURCES = $(SRCS_COMMON) generator.c

AM_CPPFLAGS = -I../common
backprop_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -I../common
backprop_LDADD = -lcuda -lcudart ../common/libutil.a

backprop_nvmgpu_CFLAGS = $(backprop_CFLAGS) -DNVMGPU
backprop_nvmgpu_LDADD = $(backprop_LDADD) $(top_srcdir)/library/src/libdragon.a $(GLIB_LIBS)
backprop_nvmgpu_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_hostreg_CFLAGS = $(backprop_CFLAGS) -DHOSTREG
backprop_hostreg_LDADD = $(backprop_LDADD)
backprop_hostreg_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_uvm_CFLAGS = $(backprop_CFLAGS) -DUVM
backprop_uvm_LDADD = $(backprop_LDADD)
backprop_uvm_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_cudamemcpy_CFLAGS = $(backprop_CFLAGS) -DCUDAMEMCPY
backprop_cudamemcpy_LDADD = $(backprop_LDADD)
backprop_cudamemcpy_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

generator_CFLAGS = $(backprop_CFLAGS) -DGENERATOR
generator_LDADD = $(backprop_LDADD)
generator_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
