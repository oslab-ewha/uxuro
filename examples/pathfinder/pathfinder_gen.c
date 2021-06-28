#include <stdio.h>
#include <stdlib.h>

#include "cudaio.h"

static void
gen_data_input(const char *folder, long size)
{
	FILE	*fp;
	char	fpath[256];
	long	i;

	snprintf(fpath, 256, "%s/data.mem", folder);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to open: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < size * size; i++) {
		int	val = rand() % 10;

		fwrite(&val, sizeof(int), 1, fp);
	}
	fclose(fp);
}

static void
gen_empty_output(const char *folder, long size)
{
	FILE	*fp;
	char	fpath[256];
	long	i;

	snprintf(fpath, 256, "%s/result.mem", folder);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to open: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < size; i++) {
		int	val = 0;

		fwrite(&val, sizeof(int), 1, fp);
	}
	fclose(fp);
}

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	long	size = *(int *)ctx;
	fprintf(fp, "%ld", size);
}

int
main(int argc, char *argv[])
{
	long	size;
	char	*folder;

	if (argc == 3) {
		size = atol(argv[1]);
		folder = argv[2];
	}
	else {
		printf("Usage: %s <each dimension size> folder\n", argv[0]);
		exit(0);
	}

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &size);

	srand(1234);

	gen_data_input(folder, size);
	gen_empty_output(folder, size);

	return 0;
}

