CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

noinst_PROGRAMS = backprop.cudamemcpy generator

backprop_cudamemcpy_SOURCES = backprop.cu facetrain.c backprop_cuda.cu ../lib/backprop.h
generator_SOURCES = generator.c backprop.cu backprop_cuda.cu ../lib/backprop.h

backprop_cudamemcpy_LDADD = ../lib/libbackprop.a -lcuda -lcudart
generator_LDADD = ../lib/libbackprop.a -lcuda -lcudart

include $(top_srcdir)/makefile.cu
