/*********************
 * renewed by cezanne@codemayo.com
 *
 * Hotspot Expand
 * by Sam Kauffman - Univeristy of Virginia
 * Generate larger input files for Hotspot by expanding smaller versions
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

#include "cudaio.h"

static void
gen_file(const char *folder, const char *fname, long rows, long cols, float val_min, float val_max)
{
	FILE	*f_out;
	char	path_out[512];
	int	i, j;

	sprintf(path_out, "%s/%s", folder, fname);
	f_out = fopen(path_out, "w+");
	if (f_out == NULL) {
		fprintf(stderr, "error: open output file: %s\n", path_out);
		exit(1);
	}

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			float	val_in = (float)((rand() / ((float)RAND_MAX / (val_max - val_min))) + val_min);
			fwrite(&val_in, sizeof(float), 1, f_out);
		}
	}

	fclose(f_out);
}

typedef struct {
	long	rows, cols;
} params_t;

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	params_t	*pparams = (params_t *)ctx;
	fprintf(fp, "%lu %lu", pparams->rows, pparams->cols);
}

int
main(int argc, char* argv[])
{
	char	*folder;
	params_t	params;

	if (argc != 4) {
		fprintf(stderr, "Usage: rows cols folder: %s\n", argv[0]);
		return 1;
	}

	params.rows = atol(argv[1]);
	params.cols = atol(argv[2]);
	folder = argv[3];

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &params);

	srand(1923);

	gen_file(folder, "power", params.rows, params.cols, 0.000017, 0.002823);
	gen_file(folder, "temperature", params.rows, params.cols, 322.980566, 343.964157);
	gen_file(folder, "output", params.rows, params.cols, 0, 0);

	fprintf(stdout, "Data written to %s\n", folder);

	return 0;
}
