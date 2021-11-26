CLEANFILES = *~
DISTCLEANFILES = *~

noinst_LIBRARIES = libutil.a

libutil_a_SOURCES = cuhelper.h cudaio.c cudaio.h timer.c timer.h
libutil_a_CFLAGS = $(CUDA_INC) -I$(top_srcdir)/library
