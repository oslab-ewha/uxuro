AUTOMAKE_OPTIONS = dejagnu
CLEANFILES = *~
DISTCLEANFILES = *~

TESTS = basictest

noinst_PROGRAMS = basictest overscription

basictest_SOURCES = basictest.cu
basictest_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

overscription_SOURCES = overscription.cu
overscription_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

AM_CPPFLAGS = -I$(top_srcdir)/library -I../../examples/common
LDADD = $(top_srcdir)/library/libuxu.a $(GLIB_LIBS) ../../examples/common/libutil.a

include $(top_srcdir)/makefile.cu
