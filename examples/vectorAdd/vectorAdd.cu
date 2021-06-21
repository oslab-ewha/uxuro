/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "cudaio.h"
#include "timer.h"
#include "cuhelper.h"

#define N_THREADS	256

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
static __global__ void
vectorAdd(const float *A, const float *B, float *C, unsigned long n_elems)
{
	unsigned long	i = (unsigned long)blockDim.x * (unsigned long)blockIdx.x + (unsigned long)threadIdx.x;

	if (i < n_elems) {
		C[i] = A[i] + B[i];
	}
}

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	unsigned	*pn_elems = (unsigned *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot option count: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%u", pn_elems) != 1) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

int
main(int argc, char *argv[])
{
	cuio_ptr_t	ptr_A, ptr_B, ptr_C;
	unsigned	n_elems, n_elems_sub = 0, i;
	unsigned	ticks_pre = 0, ticks_kern = 0, ticks_post = 0;
	const char	*folder;
	cudaError_t	err;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
		return EXIT_SUCCESS;
	}

	folder = argv[1];
	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &n_elems);

	if (getenv("N_SUB_ELEMENTS")) {
		n_elems_sub = atoi(getenv("N_SUB_ELEMENTS"));
	}

	// Print the vector length to be used, and compute its size
	printf("[Vector addition of %d elements", n_elems);
	if (n_elems_sub > 0)
		printf(" using %d sub elements", n_elems_sub);
	else
		n_elems_sub = n_elems;
	printf("]\n");

	for (i = 0; i < n_elems; i += n_elems_sub) {
		size_t	size;
		off_t	offset;

		if (i + n_elems_sub <= n_elems)
			size = n_elems_sub * sizeof(float);
		else
			size = (n_elems - i - n_elems_sub) * sizeof(float);
		offset = i * sizeof(float);

		init_tickcount();

		ptr_A = cuio_load("a.mem", offset, size, CUIO_MODE_READONLY);
		ptr_B = cuio_load("b.mem", offset, size, CUIO_MODE_READONLY);
		ptr_C = cuio_load("c.mem", offset, size, CUIO_MODE_WRITEONLY);

		cuio_memcpy_h2d(&ptr_A);
		cuio_memcpy_h2d(&ptr_B);

		ticks_pre += get_tickcount();

		// Launch the Vector Add CUDA Kernel
		int blocksPerGrid = (n_elems + N_THREADS - 1) / N_THREADS;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, N_THREADS);

		init_tickcount();
		vectorAdd<<<blocksPerGrid, N_THREADS>>>((float *)ptr_A.ptr_d, (float *)ptr_B.ptr_d, (float *)ptr_C.ptr_d, n_elems_sub);
		CUDA_CALL_SAFE(cudaDeviceSynchronize());

		err = cudaGetLastError();
		if (err != cudaSuccess)	{
			fprintf(stderr, "failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		ticks_kern += get_tickcount();

		// Copy the device result vector in device memory to the host result vector
		// in host memory.
		printf("Copy output data from the CUDA device to the host memory\n");

		init_tickcount();

		cuio_memcpy_d2h(&ptr_C);

		cuio_unload_floats("c.mem", &ptr_C);
		cuio_free_mem(&ptr_A);
		cuio_free_mem(&ptr_B);

		ticks_post += get_tickcount();
	}

	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u\n", ticks_kern);
	printf("post time(us): %u\n", ticks_post);

	return 0;
}

