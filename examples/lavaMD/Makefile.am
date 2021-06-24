CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = lavaMD.gen lavaMD

lavaMD_SOURCES = lavaMD.cu
lavaMD_gen_SOURCES = lavaMD_gen.c

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/src/libuxu.a $(GLIB_LIBS)

lavaMD_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
lavaMD_LDADD = $(common_LDADD)
lavaMD_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

lavaMD_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
lavaMD_gen_LDADD = $(common_LDADD)
lavaMD_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
