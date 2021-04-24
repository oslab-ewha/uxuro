CLEANFILES = *~
DISTCLEANFILES = *~

noinst_LIBRARIES = libdragon.a

libdragon_a_CFLAGS = $(GLIB_CFLAGS) $(CUDA_INC) -I../include

libdragon_a_SOURCES = dragon.c

include $(top_srcdir)/makefile.cu
