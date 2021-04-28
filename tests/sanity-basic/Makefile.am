CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = readtest writetest

readtest_SOURCES = read.cu
writetest_SOURCES = write.cu

readtest_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@
writetest_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

AM_CPPFLAGS = -I$(top_srcdir)/library/include
LDADD = $(top_srcdir)/library/src/libdragon.a $(GLIB_LIBS)

include $(top_srcdir)/makefile.cu
