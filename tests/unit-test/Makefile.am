AUTOMAKE_OPTIONS = dejagnu
CLEANFILES = *~
DISTCLEANFILES = *~

TESTS = basictest

noinst_PROGRAMS = ut-readonly ut-writeonly ut-readwrite ut-overscription

ut_readonly_SOURCES = ut-readonly.cu
ut_readonly_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

ut_writeonly_SOURCES = ut-writeonly.cu
ut_writeonly_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

ut_readwrite_SOURCES = ut-readwrite.cu
ut_readwrite_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

ut_overscription_SOURCES = ut-overscription.cu
ut_overscription_LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

AM_CPPFLAGS = -I$(top_srcdir)/library -I../../examples/common
LDADD = $(top_srcdir)/library/libuxu.a $(GLIB_LIBS) ../../examples/common/libutil.a

ut-readonly.o ut-writeonly ut-readwrite.o ut-overscription.o: unit_test.h

include $(top_srcdir)/makefile.cu
