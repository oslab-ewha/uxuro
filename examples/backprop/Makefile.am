CLEANFILES = *~
DISTCLEANFILES = *~

SRCS_COMMON = backprop_train.cu backprop_cuda_kernel.cuh backprop.h bpnn.c bpnn.h
SRCS = backprop.c $(SRCS_COMMON)
noinst_PROGRAMS = backprop backprop.gen

backprop_SOURCES = $(SRCS)
backprop_gen_SOURCES = $(SRCS_COMMON) generator.c

AM_CPPFLAGS = -I../common -g

common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a $(top_srcdir)/library/libuxu.a $(GLIB_LIBS)

backprop_CFLAGS = $(common_CFLAGS)
backprop_LDADD = $(common_LDADD)
backprop_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

backprop_gen_CFLAGS = $(common_CFLAGS)
backprop_gen_LDADD = $(common_LDADD)
backprop_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
