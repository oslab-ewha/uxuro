// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>

#include "cudaio.h"

#include "srad.h"

static void
gen_random_matrix(const char *folder, long size)
{
	FILE	*fp;
	char	fpath[256];
	long	i;

	snprintf(fpath, 256, "%s/matrix.mem", folder);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to create: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < size * size; i++) {
		float	val = (float)exp(rand() / (float)RAND_MAX);
		fwrite(&val, sizeof(float), 1, fp);
	}
	fclose(fp);
}

static void
gen_empty_matrix(const char *folder, const char *fname, long size)
{
	FILE	*fp;
	char	fpath[256];
	long	i;

	snprintf(fpath, 256, "%s/%s", folder, fname);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to create: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < size * size; i++) {
		float	val = 0;
		fwrite(&val, sizeof(float), 1, fp);
	}
	fclose(fp);
}

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	long	size = *(long *)ctx;
	fprintf(fp, "%ld", size);
}

int
main(int argc, char *argv[]) 
{
	long	size;
	const char	*folder;

	if (argc == 3) {
		size = atol(argv[1]);  //number of cols/rows in the domain
		if (size % 16 != 0) {
			fprintf(stderr, "size must be multiples of 16\n");
			exit(1);
		}
		folder = argv[2];
	}
	else {
		fprintf(stderr, "Usage: %s <size> <folder>\n", argv[0]);
		exit(1);
	}

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &size);

	srand(1234);

	printf("Randomizing the input matrix\n");
	gen_random_matrix(folder, size);
	gen_empty_matrix(folder, "matrix.C", size);
	gen_empty_matrix(folder, "matrix.C.E", size);
	gen_empty_matrix(folder, "matrix.C.W", size);
	gen_empty_matrix(folder, "matrix.C.N", size);
	gen_empty_matrix(folder, "matrix.C.S", size);

	return 0;
}
