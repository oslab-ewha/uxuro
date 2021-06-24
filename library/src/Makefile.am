CLEANFILES = *~
DISTCLEANFILES = *~

noinst_LIBRARIES = libuxu.a

libuxu_a_CFLAGS = $(GLIB_CFLAGS) $(CUDA_INC) -I../include

libuxu_a_SOURCES = libuxu.c

include $(top_srcdir)/makefile.cu
