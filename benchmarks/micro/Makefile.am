CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = mb_advise

mb_advise_SOURCES = mb_advise.cu ../benchmark.h
mb_advise_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

AM_CPPFLAGS = -I..

mb_advise.o: ../benchmark.h

include $(top_srcdir)/makefile.cu
