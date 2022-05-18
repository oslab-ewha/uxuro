#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define ERROR(args...)	\
	do {			\
		char errmsg[1024];	\
		snprintf(errmsg, 1024, args);	\
		fprintf(stderr, "ERROR: %s\n", errmsg);	\
		exit(1);				\
	} while (0)

#define CUDA_CHECK(stmt, errmsg)		\
	do {					\
	        cudaError_t err = stmt;		\
	        if (err != cudaSuccess)	{	\
			ERROR("%s: %s", errmsg, cudaGetErrorString(err)); \
		}				\
	} while (0)

#endif
