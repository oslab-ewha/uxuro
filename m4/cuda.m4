##### 
#
# SYNOPSIS
#
# AX_CHECK_CUDA
#
# DESCRIPTION
#
# Figures out if CUDA Driver API/nvcc is available, i.e. existence of:
# 	cuda.h
#   libcuda.so
#   nvcc
#
# If something isn't found, fails straight away.
#
# Locations of these are included in 
#   CUDA_CPPFLAGS and 
#   CUDA_LDFLAGS.
# Path to nvcc is included as
#   NVCC_PATH
# in config.h
# 
# The author is personally using CUDA such that the .cu code is generated
# at runtime, so don't expect any automake magic to exist for compile time
# compilation of .cu files.
#
# LICENCE
# Public domain
#
# AUTHOR
# wili
#
##### 

AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this
AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])
AC_ARG_WITH(microarch, [  --with-microarch=<arch name>         GPU micro architecture name: pascal, volta], [cuda_microarch=$withval])

# Setting the prefix to the default if only --with-cuda was given
if test "$cuda_prefix" == "yes"; then
	if test "$withval" == "yes"; then
		cuda_prefix="/usr/local/cuda"
	fi
fi

if test -n "$cuda_microarch"; then
	case $cuda_microarch in
		fermi)
		NVCC_ARCHITECTURE="--gpu-architecture sm_20"
		NVCC_ARCHITECTURE=""
		AC_DEFINE([CUDA_COMPUTE], [20], [GPU micro architecture])
		;;
		maxwell)
		NVCC_ARCHITECTURE="--gpu-architecture sm_53"
		AC_DEFINE([CUDA_COMPUTE], [53], [GPU micro architecture])
		;;
		p100)
		NVCC_ARCHITECTURE="--gpu-architecture sm_60"
		AC_DEFINE([CUDA_COMPUTE], [60], [GPU micro architecture])
		;;
		pascal)
		NVCC_ARCHITECTURE="--gpu-architecture sm_61"
		AC_DEFINE([CUDA_COMPUTE], [61], [GPU micro architecture])
		;;
		volta)
		NVCC_ARCHITECTURE="--gpu-architecture sm_70"
		AC_DEFINE([CUDA_COMPUTE], [70])
		;;
		*)
		AC_MSG_ERROR([invalid micro architecture: $cuda_microarch], 1)
		;;
	esac
fi

# Checking for nvcc
AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
if test -x "$cuda_prefix/bin/nvcc"; then
	AC_MSG_RESULT([found])
	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	AC_SUBST([NVCC_PATH])
	NVCC_PATH="$cuda_prefix/bin/nvcc"
else
	AC_MSG_RESULT([not found!])
	AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
fi

# We need to add the CUDA search directories for header and lib searches

# Saving the current flags
ax_save_CPPFLAGS="${CPPFLAGS}"
ax_save_LDFLAGS="${LDFLAGS}"
ax_save_LIBS="${LIBS}"

# Announcing the new variables
AC_SUBST([NVCC_CPPFLAGS])
AC_SUBST([NVCC_LDFLAGS])
AC_SUBST([CUDA_INC])
AC_SUBST([CUDA_LIBS])
AC_SUBST([NVCC_ARCHITECTURE])

NVCC_CPPFLAGS="-I$cuda_prefix/include $NVCC_CPPFLAGS"
CPPFLAGS="$NVCC_CPPFLAGS $CPPFLAGS"
NVCC_LDFLAGS="-L$cuda_prefix/lib64 -L$cuda_prefix/lib64/stubs $NVCC_LDFLAGS"
LDFLAGS="$NVCC_LDFLAGS $LDFLAGS"
CUDA_INC="-I$cuda_prefix/include"
CUDA_LIBS="-lcuda -lcudart $CUDA_LIBS"
LIBS="$CUDA_LIBS $LIBS"

#CUDA_LIBS= # the libs we specify manually for each target

AM_CONDITIONAL([HAVE_CUDA_H], false)
AM_CONDITIONAL([HAVE_CUDA_CUSOLVERDN_H], false)

# And the header and the lib
AC_CHECK_HEADER([cuda.h], 
                [
                  AC_DEFINE([HAVE_CUDA_H],[1],[defined if cuda.h is found.])
                  AM_CONDITIONAL([HAVE_CUDA_H], true)
                ], 
                AC_MSG_WARN([Couldn't find cuda.h. Install from https://developer.nvidia.com/cuda-downloads]), 
                [#include <cuda.h>]
               )
AC_CHECK_LIB([cuda],
             [cuInit],
             AC_MSG_WARN([In case of problems set LD_LIBRARY_PATH so libcuda.so is found]), 
             AC_MSG_WARN([Couldn't find libcuda.so]))

# Returning to the original flags
CPPFLAGS=${ax_save_CPPFLAGS}
LDFLAGS=${ax_save_LDFLAGS}
LIBS=${ax_save_LIBS}
])
