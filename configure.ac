#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([uxu], [0.1.0], [cezanne@codemayo.com])
AC_CONFIG_SRCDIR([library/libuxu.c])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([foreign])

AM_SILENT_RULES([yes])

AM_MAINTAINER_MODE([disable])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AX_CHECK_CUDA

PKG_CHECK_MODULES([GLIB], [glib-2.0])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h strings.h sys/time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

AC_DEFINE(_GNU_SOURCE, [], [GNU Supported Code])

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([gettimeofday sqrt strchr])

AC_CONFIG_FILES([Makefile
                 library/Makefile
		 tools/Makefile
		 tools/kmsglog/Makefile
                 examples/Makefile
                 examples/common/Makefile
		 examples/BlackScholes/Makefile
		 examples/backprop/Makefile
		 examples/binomialOptions/Makefile
		 examples/hotspot/Makefile
		 examples/lavaMD/Makefile
		 examples/pathfinder/Makefile
		 examples/srad_v2/Makefile
		 examples/vectorAdd/Makefile
		 tests/Makefile
		 tests/sanity-basic/Makefile
		 tests/unit-test/Makefile
		 ])

AC_OUTPUT
