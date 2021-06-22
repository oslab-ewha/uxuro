/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "binomialOptions_common.h"
#include "realtype.h"

#include "binomialOptions_kernel.cu"

#include "cudaio.h"
#include "timer.h"
#include "cuhelper.h"

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	unsigned	*pn_options = (unsigned *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot get # of boxes: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%u", pn_options) != 1) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

int
main(int argc, char *argv[])
{
	cuio_ptr_t	ptr_option_data, ptr_calls;
	unsigned	n_options;
	unsigned	ticks_pre, ticks_kern, ticks_post;
	const char	*folder;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	folder = argv[1];

	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &n_options);

	init_tickcount();

	ptr_option_data = cuio_load("optionData.mem", 0, sizeof(option_data_t) * n_options, CUIO_MODE_READONLY);
	ptr_calls = cuio_load_floats("callValue.mem", n_options, CUIO_MODE_WRITEONLY);

	cuio_memcpy_h2d(&ptr_option_data);

	checkCudaErrors(cudaDeviceSynchronize());

	ticks_pre = get_tickcount();

	printf("Running GPU binomial tree...\n");

	init_tickcount();

	binomialOptionsKernel<<<n_options, THREADBLOCK_SIZE>>>((option_data_t *)ptr_option_data.ptr_d, (real *)ptr_calls.ptr_d);
	checkCudaErrors(cudaDeviceSynchronize());
	ticks_kern = get_tickcount();

	getLastCudaError("binomialOptionsKernel() execution failed.\n");

	init_tickcount();
	cuio_memcpy_d2h(&ptr_calls);
	cuio_unload_floats("callValue.mem", &ptr_calls);
	cuio_free_mem(&ptr_option_data);
	ticks_post = get_tickcount();

	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u\n", ticks_kern);
	printf("post time(us): %u\n", ticks_post);

	return 0;
}
