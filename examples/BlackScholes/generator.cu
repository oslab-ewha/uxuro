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

#include "cudaio.h"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
static float
RandFloat(float low, float high)
{
	float t = (float)rand() / (float)RAND_MAX;
	return (1.0f - t) * low + t * high;
}

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	fprintf(fp, "%lu", *(long *)ctx);
}

int
main(int argc, char *argv[])
{
	long	opt_n, opt_size;
	char	*folder;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <OPT_N> <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	opt_n = atol(argv[1]);
	opt_size = opt_n * sizeof(float);
	folder = argv[2];

	cuio_init(CUIO_TYPE_HOST, folder, 1);

	// Start logs
	printf("[%s] - Starting...\n", argv[0]);

	//'h_' prefix - CPU (host) memory space
	cuio_ptr_t	stockPrice, optionStrike, optionYears;

	long	i;

	printf("Initializing data...\n");
	printf("...allocating CPU memory for options.\n");
	stockPrice = cuio_alloc_mem(opt_size);
	optionStrike = cuio_alloc_mem(opt_size);
	optionYears = cuio_alloc_mem(opt_size);

	printf("...generating input data in CPU mem.\n");
	srand(5347);

	//Generate options set
	for (i = 0; i < opt_n; i++) {
		CUIO_FLOATS_ITEM(stockPrice, i) = RandFloat(5.0f, 30.0f);
		CUIO_FLOATS_ITEM(optionStrike, i) = RandFloat(1.0f, 100.0f);
		CUIO_FLOATS_ITEM(optionYears, i) = RandFloat(0.25f, 10.0f);
	}

	cuio_unload_floats("StockPrice.mem", &stockPrice);
	cuio_unload_floats("OptionStrike.mem", &optionStrike);
	cuio_unload_floats("OptionYears.mem", &optionYears);

	cuio_save_conf(confer_save, &opt_n);

	return 0;
}
