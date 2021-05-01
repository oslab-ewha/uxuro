#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "backprop.h"

#include "bpnn.h"
#include "cudaio.h"

static void
randomize_floats(float *data, int count)
{
	int	i;

	for (i = 0; i < count; i++) {
		data[i] = (float)rand() / RAND_MAX;
	}
}

static void
randomize_output(BPNN *net)
{
	long	i;
	for (i = 0; i <= net->output_n; i++) {
		//w[i] = (float)rand() / RAND_MAX;
		net->target[i] = 0.1;
	}
}

static void
randomize_bpnn(BPNN *net)
{
	randomize_floats(net->input_units, net->input_n + 1);
	randomize_floats(net->input_weights, (net->input_n + 1) * (net->hidden_n + 1));
	randomize_floats(net->hidden_weights, (net->hidden_n + 1) * (net->output_n + 1));
	randomize_output(net);
}

static void
save_net_conf(BPNN *net, const char *folder)
{
	FILE	*fp;
	char	fpath[256];
	char	buf[1024];

	snprintf(fpath, 256, "%s/net.conf", folder);
	fp = fopen(fpath, "w");
	if (fp == NULL) {
		fprintf(stderr, "cannot open for write: %s\n", fpath);
		exit(2);
	}
	fprintf(fp, "%ld %ld %ld", net->input_n, net->hidden_n, net->output_n);
	fclose(fp);
}

static void
generate_bpnn(const char *folder, int n_inp)
{
	BPNN	*net;
	int	n_hid = 16, n_out = 1;
	char	fpath[256];

	srand(7);

	net = bpnn_create(n_inp, n_hid, n_out);

	net->input_units = cuio_load_floats(NULL, n_inp + 1, CUIO_MODE_NONE);
	net->target = cuio_load_floats(NULL, n_out + 1, CUIO_MODE_NONE);
	net->input_weights = cuio_load_floats(NULL, (n_inp + 1) * (n_hid + 1), CUIO_MODE_NONE);
	net->hidden_weights = cuio_load_floats(NULL, (n_hid + 1) * (n_out + 1), CUIO_MODE_NONE);

	randomize_bpnn(net);

	snprintf(fpath, 256, "%s/input_units.mem", folder);
	cuio_unload_floats(fpath, net->input_n + 1, net->input_units, CUIO_MODE_WRITEONLY);
	snprintf(fpath, 256, "%s/target.mem", folder);
	cuio_unload_floats(fpath, net->output_n + 1, net->target, CUIO_MODE_WRITEONLY);
	snprintf(fpath, 256, "%s/input_weights.mem", folder);
	cuio_unload_floats(fpath, (net->input_n + 1) * (net->hidden_n + 1), net->input_weights, CUIO_MODE_WRITEONLY);
	snprintf(fpath, 256, "%s/hidden_weights.mem", folder);
	cuio_unload_floats(fpath, (net->hidden_n + 1) * (net->output_n + 1), net->hidden_weights, CUIO_MODE_WRITEONLY);

	save_net_conf(net, folder);
	bpnn_free(net);
}

static void
check_folder(const char *folder)
{
	if (access(folder, F_OK) == 0) {
		fprintf(stderr, "folder exist: %s\n", folder);
		fprintf(stderr, "You should provide a non-existent folder\n");
		exit(1);
	}
	if (mkdir(folder, 0700) < 0) {
		fprintf(stderr, "cannot make directory: %s\n", folder);
		exit(2);
	}
}

int
main(int argc, char *argv[])
{
	long	n_inp;
	char	*folder;

	if (argc != 3) {
		fprintf(stderr, "usage: %s <num of input elements> <folder>\n", argv[0]);
		return 1;
	}

	n_inp = atol(argv[1]);

	folder = argv[2];
	n_inp = (n_inp / 16) * 16;

	printf("input size: %ld\n", n_inp);

	check_folder(folder);

	generate_bpnn(folder, n_inp);

	return 0;
}
