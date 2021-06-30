#include <stdio.h>
#include <stdlib.h>

#include "cudaio.h"

static const char	*folder;

static void
make_vector_file(const char *fname, long n_elems)
{
	char	fpath[512];
	FILE	*fp;
	long	i;

	snprintf(fpath, 512, "%s/%s", folder, fname);
	if ((fp = fopen(fpath, "w+")) == NULL) {
		fprintf(stderr, "failed to create file: %s\n", fpath);
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < n_elems; i++) {
		float	val;

		val = (float)rand();
		fwrite(&val, sizeof(float), 1, fp);
	}
	fclose(fp);
}

static void
make_empty_file(const char *fname, long n_elems)
{
	FILE	*fp;
	char	fpath[256];
	long	i;

	snprintf(fpath, 256, "%s/%s", folder, fname);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to create: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < n_elems; i++) {
		float	val = 0;
		fwrite(&val, sizeof(float), 1, fp);
	}
	fclose(fp);
}

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	long	n_elems = *(long *)ctx;
	fprintf(fp, "%ld", n_elems);
}

int 
main(int argc, char *argv [])
{
	long	n_elems;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <n elements> <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	n_elems = atol(argv[1]);
	folder = argv[2];

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &n_elems);

	// Print configuration
	printf("# of elements = %ld\n", n_elems);

	srand(1234);

	make_vector_file("a.mem", n_elems);
	make_vector_file("b.mem", n_elems);
	make_empty_file("c.mem", n_elems);

	return 0;
}
