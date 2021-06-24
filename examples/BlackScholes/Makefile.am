CLEANFILES = *~
DISTCLEANFILES = *~

SRCS_COMMON = BlackScholes_kernel.cuh
SRCS = BlackScholes.cu $(SRCS_COMMON)
noinst_PROGRAMS = BlackScholes.gen BlackScholes

BlackScholes_SOURCES = $(SRCS)
BlackScholes_gen_SOURCES = generator.cu $(SRCS_COMMON)

AM_CPPFLAGS = -I../common
common_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library/include -I../common
common_LDADD = -lcuda -lcudart ../common/libutil.a ../common/libutil.a $(top_srcdir)/library/src/libuxu.a $(GLIB_LIBS)

BlackScholes_CFLAGS = $(common_CFLAGS) -DCUDAMEMCPY
BlackScholes_LDADD = $(common_LDADD)
BlackScholes_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

BlackScholes_gen_CFLAGS = $(common_CFLAGS) -DGENERATOR
BlackScholes_gen_LDADD = $(common_LDADD)
BlackScholes_gen_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

include $(top_srcdir)/makefile.cu
