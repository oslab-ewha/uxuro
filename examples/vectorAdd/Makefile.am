CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = vectorAdd.gen vectorAdd

vectorAdd_SOURCES = vectorAdd.cu
vectorAdd_gen_SOURCES = vectorAdd_gen.c

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/src/libdragon.a $(GLIB_LIBS)

vectorAdd_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
vectorAdd_LDADD = $(common_LDADD)
vectorAdd_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

vectorAdd_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
vectorAdd_gen_LDADD = $(common_LDADD)
vectorAdd_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
