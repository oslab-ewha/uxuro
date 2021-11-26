CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = srad.gen srad

srad_SOURCES = srad.cu
srad_gen_SOURCES = srad_gen.c

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/libuxu.a $(GLIB_LIBS)

srad_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
srad_LDADD = $(common_LDADD)
srad_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

srad_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
srad_gen_LDADD = $(common_LDADD)
srad_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
