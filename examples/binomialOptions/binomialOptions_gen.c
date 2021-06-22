#include <stdio.h>
#include <stdlib.h>

#include "binomialOptions_common.h"
#include "realtype.h"

#include "cudaio.h"

static real
randData(real low, real high)
{
	real t = (real)rand() / (real)RAND_MAX;
	return ((real)1.0 - t) * low + t * high;
}

static void
gen_option_data(const char *folder, unsigned n_options)
{
	FILE	*fp;
	char	fpath[256];
	unsigned	i;

	snprintf(fpath, 256, "%s/optionData.mem", folder);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to create: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < n_options; i++) {
		option_data_t	od;

		od.S = randData(5.0f, 30.0f);
		od.X = randData(1.0f, 100.0f);
		od.vDt = randData(0.25f, 10.0f);
		od.puByDf = 0.06f;
		od.pdByDf = 0.10f;

		fwrite(&od, sizeof(option_data_t), 1, fp);
	}
	fclose(fp);
}

static void
gen_empty_output(const char *folder, unsigned n_options)
{
	FILE	*fp;
	char	fpath[256];
	unsigned	i;

	snprintf(fpath, 256, "%s/callValue.mem", folder);
	if ((fp = fopen(fpath, "w+")) == 0) {
		fprintf(stderr, "failed to open: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < n_options; i++) {
		int	val = 0;

		fwrite(&val, sizeof(int), 1, fp);
	}
	fclose(fp);
}

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	unsigned	n_options = *(unsigned *)ctx;
	fprintf(fp, "%u", n_options);
}

int
main(int argc, char *argv[])
{
	unsigned	n_options;
	char	*folder;

	if (argc == 3) {
		n_options = atol(argv[1]);
		folder = argv[2];
	}
	else {
		printf("Usage: %s <# of optinos> folder\n", argv[0]);
		exit(0);
	}

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &n_options);

	srand(1234);

	printf("Generating option data...\n");
	gen_option_data(folder, n_options);
	gen_empty_output(folder, n_options);

	return 0;
}
