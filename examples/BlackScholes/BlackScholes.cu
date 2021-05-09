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
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "cudaio.h"
#include "timer.h"

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

const long  NUM_ITERATIONS = 1;


const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	long	*popt_n = (long *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot option count: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%ld", popt_n) != 1) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

int
main(int argc, char *argv[])
{
	long	opt_n;
	char	*folder;
	cuio_ptr_t	stockPrice, optionStrike, optionYears;
	cuio_ptr_t	callResult, putResult;
	unsigned	ticks_pre, ticks_kern, ticks_post;
	float		resCall1, resCall2, resPut1, resPut2;
	long	i;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	folder = argv[1];

	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &opt_n);

	printf("Initializing data...\n");

	init_tickcount();
	callResult = cuio_load_floats("CallResult.mem", opt_n, CUIO_MODE_WRITEONLY);
	putResult = cuio_load_floats("PutResult.mem", opt_n, CUIO_MODE_WRITEONLY);
	stockPrice = cuio_load_floats("StockPrice.mem", opt_n, CUIO_MODE_READONLY);
	optionStrike = cuio_load_floats("OptionStrike.mem", opt_n, CUIO_MODE_READONLY);
	optionYears = cuio_load_floats("OptionYears.mem", opt_n, CUIO_MODE_READONLY);

	cuio_memset_d(&callResult, 0);
	cuio_memset_d(&putResult, 0);

	cuio_memcpy_h2d(&stockPrice);
	cuio_memcpy_h2d(&optionStrike);
	cuio_memcpy_h2d(&optionYears);

	checkCudaErrors(cudaDeviceSynchronize());

	ticks_pre = get_tickcount();

	printf("Executing Black-Scholes GPU kernel (%li iterations)...\n", NUM_ITERATIONS);

	init_tickcount();

	for (i = 0; i < NUM_ITERATIONS; i++) {
		BlackScholesGPU<<<DIV_UP((opt_n/2), 128), 128/*480, 128*/>>>(
			(float2 *)CUIO_FLOATS_D(callResult), (float2 *)CUIO_FLOATS_D(putResult),
			(float2 *)CUIO_FLOATS_D(stockPrice), (float2 *)CUIO_FLOATS_D(optionStrike), (float2 *)CUIO_FLOATS_D(optionYears),
			RISKFREE, VOLATILITY, opt_n);
		getLastCudaError("BlackScholesGPU() execution failed\n");
		checkCudaErrors(cudaDeviceSynchronize());
	}

	checkCudaErrors(cudaDeviceSynchronize());
	ticks_kern = get_tickcount();

	printf("\nReading back GPU results...\n");
	//Read back GPU results to compare them to CPU results

	init_tickcount();

	cuio_memcpy_d2h(&callResult);
	cuio_memcpy_d2h(&putResult);

	resCall1 = CUIO_FLOATS_ITEM(callResult, 0);
	resCall2 = CUIO_FLOATS_ITEM(callResult, opt_n - 1);
	resPut1 = CUIO_FLOATS_ITEM(callResult, 0);
	resPut2 = CUIO_FLOATS_ITEM(callResult, opt_n - 1);

	cuio_unload_floats("CallResult.mem", &callResult);
	cuio_unload_floats("PutResult.mem", &putResult);

	cuio_free_mem(&optionYears);
	cuio_free_mem(&optionStrike);
	cuio_free_mem(&stockPrice);

	ticks_post = get_tickcount();

	printf("Result: Call(%f,%f), Put(%f,%f)\n", resCall1, resCall2, resPut1, resPut2);

	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u\n", ticks_kern);
	printf("post time(us): %u\n", ticks_post);

	exit(EXIT_SUCCESS);
}
