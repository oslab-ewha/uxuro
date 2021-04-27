/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 * Totally modified by cezanne(cezanne@codemayo.com)
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#include "backprop.h"

long	layer_size = 0;
char	*folder;

#ifdef GENERATOR
int generate_bpnn(long layer_size);
#else
int backprop_train(void);
#endif

int
main(int argc, char *argv[])
{
	long seed;

	if (argc != 3) {
		fprintf(stderr, "usage: %s <num of input elements> <folder>\n", argv[0]);
		return 1;
	}

	layer_size = atol(argv[1]);

	folder = argv[2];
	if (layer_size % 16 != 0) {
		fprintf(stderr, "The number of input points must be divided by 16\n");
		return 2;
	}

	seed = 7;
	printf("Random number generator seed: %ld\n", seed);
	srand(seed);

#ifdef GENERATOR
	return generate_bpnn(layer_size);
#else
	return backprop_train();
#endif
}
