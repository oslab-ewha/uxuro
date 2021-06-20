#include <stdio.h>
#include <stdlib.h>

#include "cudaio.h"

static const char	*folder;

static void
make_vector_file(const char *fname, int n_elems)
{
	char	fpath[512];
	FILE	*fp;
	int	i;

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
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	int	n_elems = *(int *)ctx;
	fprintf(fp, "%u", n_elems);
}

int 
main(int argc, char *argv [])
{
	int	n_elems;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <n elements> <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	n_elems = atoi(argv[1]);
	folder = argv[2];

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &n_elems);

	// Print configuration
	printf("# of elements = %d\n", n_elems);

	srand(1234);

	make_vector_file("a.mem", n_elems);
	make_vector_file("b.mem", n_elems);

	return 0;
}
