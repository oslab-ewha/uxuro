#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>

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
		CUIO_FLOATS_ITEM(net->target, i) = 0.1;
	}
}

static void
randomize_bpnn(BPNN *net)
{
	randomize_floats(CUIO_FLOATS_H(net->input_units), net->input_n + 1);
	randomize_floats(CUIO_FLOATS_H(net->input_weights), (net->input_n + 1) * (net->hidden_n + 1));
	randomize_floats(CUIO_FLOATS_H(net->hidden_weights), (net->hidden_n + 1) * (net->output_n + 1));
	randomize_output(net);
}

static void
zero_prev_weights(BPNN *net)
{
	memset(net->input_prev_weights.ptr_h, 0, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1));
	memset(net->hidden_prev_weights.ptr_h, 0, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1));
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

	srand(7);

	net = bpnn_create(n_inp, n_hid, n_out);

	net->input_units = cuio_alloc_mem((n_inp + 1) * sizeof(float));
	net->target = cuio_alloc_mem((n_out + 1) * sizeof(float));
	net->input_weights = cuio_alloc_mem((n_inp + 1) * (n_hid + 1) * sizeof(float));
	net->hidden_weights = cuio_alloc_mem((n_hid + 1) * (n_out + 1) * sizeof(float));
	randomize_bpnn(net);

	net->input_prev_weights = cuio_alloc_mem((n_inp + 1) * (n_hid + 1) * sizeof(float));
	net->hidden_prev_weights = cuio_alloc_mem((n_hid + 1) * (n_out + 1) * sizeof(float));
	zero_prev_weights(net);

	cuio_unload_floats("input_units.mem", &net->input_units);
	cuio_unload_floats("target.mem", &net->target);
	cuio_unload_floats("input_weights.mem", &net->input_weights);
	cuio_unload_floats("hidden_weights.mem", &net->hidden_weights);
	cuio_unload_floats("input_weights.prev.mem", &net->input_prev_weights);
	cuio_unload_floats("hidden_weights.prev.mem", &net->hidden_prev_weights);

	save_net_conf(net, folder);
	bpnn_free(net);
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

	cuio_init(CUIO_TYPE_HOST, folder, 1);
	generate_bpnn(folder, n_inp);

	return 0;
}
