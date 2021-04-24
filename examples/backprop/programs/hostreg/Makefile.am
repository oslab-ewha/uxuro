CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

noinst_PROGRAMS = backprop.hostreg

backprop_hostreg_SOURCES = backprop.cu facetrain.c backprop_cuda.cu ../lib/backprop.h

backprop_hostreg_LDADD = ../lib/libbackprop.a -lcuda -lcudart

include $(top_srcdir)/makefile.cu
