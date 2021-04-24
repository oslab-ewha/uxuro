CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

noinst_PROGRAMS = backprop.nvmgpu

backprop_nvmgpu_SOURCES = backprop.cu facetrain.c backprop_cuda.cu ../lib/backprop.h

backprop_nvmgpu_LDADD = ../lib/libbackprop.a -lcuda -lcudart $(top_srcdir)/library/src/libdragon.a $(GLIB_LIBS)

include $(top_srcdir)/makefile.cu
