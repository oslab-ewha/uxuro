CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = pathfinder.gen pathfinder

pathfinder_SOURCES = pathfinder.cu
pathfinder_gen_SOURCES = pathfinder_gen.c

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/src/libuxu.a $(GLIB_LIBS)

pathfinder_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
pathfinder_LDADD = $(common_LDADD)
pathfinder_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

pathfinder_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
pathfinder_gen_LDADD = $(common_LDADD)
pathfinder_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
