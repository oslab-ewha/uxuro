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

int backprop_train(const char *folder);

int
main(int argc, char *argv[])
{
	char	*folder;

	if (argc != 2) {
		fprintf(stderr, "usage: %s <folder>\n", argv[0]);
		return 1;
	}

	folder = argv[1];
	cuio_init(CUIO_TYPE_NONE, folder, 0);
	return backprop_train(folder);
}
