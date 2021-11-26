CLEANFILES = *~
DISTCLEANFILES = *~

noinst_PROGRAMS = readtest writetest

readtest_SOURCES = read.cu
writetest_SOURCES = write.cu

readtest_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@
writetest_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

AM_CPPFLAGS = -I$(top_srcdir)/library -I../../examples/common
LDADD = $(top_srcdir)/library/libuxu.a $(GLIB_LIBS) ../../examples/common/libutil.a

include $(top_srcdir)/makefile.cu
