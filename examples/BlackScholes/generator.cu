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

static void
gen_inputs(long n_inputs)
{
	cuio_ptr_t	stockPrice, optionStrike, optionYears;
	long	input_size = n_inputs * sizeof(float);
	long	i;

	stockPrice = cuio_alloc_mem(input_size);
	optionStrike = cuio_alloc_mem(input_size);
	optionYears = cuio_alloc_mem(input_size);

	//Generate options set
	for (i = 0; i < n_inputs; i++) {
		CUIO_FLOATS_ITEM(stockPrice, i) = RandFloat(5.0f, 30.0f);
		CUIO_FLOATS_ITEM(optionStrike, i) = RandFloat(1.0f, 100.0f);
		CUIO_FLOATS_ITEM(optionYears, i) = RandFloat(0.25f, 10.0f);
	}

	cuio_unload_floats("StockPrice.mem", &stockPrice);
	cuio_unload_floats("OptionStrike.mem", &optionStrike);
	cuio_unload_floats("OptionYears.mem", &optionYears);
}

#define N_BATCH_MAX	500000000

int
main(int argc, char *argv[])
{
	long	n_opts;
	char	*folder;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <# of options> <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	n_opts = atol(argv[1]);
	folder = argv[2];

	cuio_init(CUIO_TYPE_GENERATOR, folder);

	printf("...generating BlackScholes input data: # Option: %ld\n", n_opts);

	srand(5347);

	while (n_opts > 0) {
		long	n_inputs = n_opts > N_BATCH_MAX ? N_BATCH_MAX: n_opts;

		gen_inputs(n_inputs);
		n_opts -= n_inputs;
	}

	cuio_save_conf(confer_save, &n_opts);
	printf("done\n");
	return 0;
}
