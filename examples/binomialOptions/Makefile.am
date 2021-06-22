CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = binomialOptions.gen binomialOptions

binomialOptions_SOURCES = binomialOptions.cu
binomialOptions_gen_SOURCES = binomialOptions_gen.c

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/src/libdragon.a $(GLIB_LIBS)

binomialOptions_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
binomialOptions_LDADD = $(common_LDADD)
binomialOptions_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

binomialOptions_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
binomialOptions_gen_LDADD = $(common_LDADD)
binomialOptions_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
