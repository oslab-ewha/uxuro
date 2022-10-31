CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = mb_advise mb_relay mb_vectoradd mb_matrixmul

mb_advise_SOURCES = mb_advise.cu mb_common.cu ../benchmark.h
mb_relay_SOURCES = mb_relay.cu mb_common.cu ../benchmark.h
mb_vectoradd_SOURCES = mb_vectoradd.cu mb_common.cu ../benchmark.h
mb_matrixmul_SOURCES = mb_matrixmul.cu mb_common.cu ../benchmark.h

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@
LDADD = $(top_srcdir)/examples/common/libutil.a -lcuda

AM_CPPFLAGS = -I.. -I$(top_srcdir)/examples/common

mb_advise.o mb_relay.o mb_vectoradd.o mb_matrixmul.o : ../benchmark.h

include $(top_srcdir)/makefile.cu
