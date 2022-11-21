CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = mb_advise mb_relay mb_vectoradd mb_matrixmul mb_stream mb_copy mb_random mb_compute_pi

mb_advise_SOURCES = mb_advise.cu mb_common.cu ../benchmark.h
mb_relay_SOURCES = mb_relay.cu mb_common.cu ../benchmark.h
mb_vectoradd_SOURCES = mb_vectoradd.cu mb_common.cu ../benchmark.h
mb_matrixmul_SOURCES = mb_matrixmul.cu mb_common.cu ../benchmark.h
mb_stream_SOURCES = mb_stream.cu mb_common.cu ../benchmark.h
mb_copy_SOURCES = mb_copy.cu mb_common.cu ../benchmark.h
mb_random_SOURCES = mb_random.cu mb_common.cu ../benchmark.h
mb_compute_pi_SOURCES = mb_compute_pi.cu mb_common.cu ../benchmark.h

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@
LDADD = $(top_srcdir)/examples/common/libutil.a -lcuda

AM_CPPFLAGS = -I.. -I$(top_srcdir)/examples/common

mb_advise.o mb_relay.o mb_vectoradd.o mb_matrixmul.o mb_stream.o mb_random.o mb_compute_pi.o : ../benchmark.h

include $(top_srcdir)/makefile.cu
