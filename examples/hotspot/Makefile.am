CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = hotspot.gen hotspot

hotspot_SOURCES = hotspot.cu
hotspot_gen_SOURCES = hotspot_gen.cpp

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/include -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/libuxu.a $(GLIB_LIBS)

hotspot_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
hotspot_LDADD = $(common_LDADD)
hotspot_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

hotspot_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
hotspot_gen_LDADD = $(common_LDADD)
hotspot_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
