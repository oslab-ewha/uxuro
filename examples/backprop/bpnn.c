#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "backprop.h"
#include "cudaio.h"

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

extern void store_data(const char *fname, size_t count, float *data);
extern void free_data(float *data);

static float
squash(float x)
{
	//x = -x;
	//m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
	//return(1.0 / (1.0 + m));
	return (1.0 / (1.0 + exp(-x)));
}

void
bpnn_prepare(BPNN *net, unsigned long num_blocks)
{
	cuio_memcpy_h2d(&net->input_units);
	cuio_memcpy_h2d(&net->input_weights);
	net->partial_sum = cuio_alloc_mem(num_blocks * WIDTH * sizeof(float));
}

void
bpnn_update_hidden(BPNN *net, unsigned long num_blocks)
{
	long	j;

	cuio_memcpy_d2h(&net->partial_sum);

	for (j = 1; j <= net->hidden_n; j++) {
		double	sum;
		long	k;

		sum = 0.0;
		for (k = 0; k < num_blocks; k++) {
			sum += CUIO_FLOATS_ITEM(net->partial_sum, k * net->hidden_n + j - 1);
		}
		sum += CUIO_FLOATS_ITEM(net->input_weights, j);
		CUIO_FLOATS_ITEM(net->hidden_units, j) = (float)(1.0 / (1.0 + exp(-sum)));
	}
}

void
bpnn_layerforward(BPNN *net)
{
	float	sum;
	long	j, k;

	/*** Set up thresholding unit ***/
	CUIO_FLOATS_ITEM(net->hidden_units, 0) = 1.0;
	/*** For each unit in second layer ***/
	for (j = 1; j <= net->output_n; j++) {
		/*** Compute weighted sum of its inputs ***/
		sum = 0.0;
		for (k = 0; k <= net->hidden_n; k++) {	
			sum += CUIO_FLOATS_ITEM(net->hidden_weights, k * (net->output_n + 1) + j) * CUIO_FLOATS_ITEM(net->hidden_units, k); 
		}
		CUIO_FLOATS_ITEM(net->output_units, j) = squash(sum);
	}
}

float
bpnn_output_error(BPNN *net)
{
	long	j;
	float	o, t, errsum;

	errsum = 0.0;
	for (j = 1; j <= net->output_n; j++) {
		o = CUIO_FLOATS_ITEM(net->output_units, j);
		t = CUIO_FLOATS_ITEM(net->target, j);
		CUIO_FLOATS_ITEM(net->output_delta, j) = o * (1.0 - o) * (t - o);
		errsum += ABS(CUIO_FLOATS_ITEM(net->output_delta, j));
	}
	return errsum;
}

float
bpnn_hidden_error(BPNN *net)
{
	long	j, k;
	float	h, sum, errsum;

	errsum = 0.0;
	for (j = 1; j <= net->hidden_n; j++) {
		h = CUIO_FLOATS_ITEM(net->hidden_units, j);
		sum = 0.0;
		for (k = 1; k <= net->output_n; k++) {
			sum += CUIO_FLOATS_ITEM(net->output_delta, k) * CUIO_FLOATS_ITEM(net->hidden_weights, j * (net->output_n + 1) + k);
		}
		CUIO_FLOATS_ITEM(net->hidden_delta, j) = h * (1.0 - h) * sum;
		errsum += ABS(CUIO_FLOATS_ITEM(net->hidden_delta, j));
	}
	return errsum;
}

void
bpnn_adjust_weights(BPNN *net)
{
	float	new_dw;
	long	k, j;

	CUIO_FLOATS_ITEM(net->hidden_units, 0) = 1.0;
	//eta = 0.3;
	//momentum = 0.3;

	for (j = 1; j <= net->output_n; j++) {
		for (k = 0; k <= net->hidden_n; k++) {
			new_dw = ((ETA * CUIO_FLOATS_ITEM(net->output_delta, j) * CUIO_FLOATS_ITEM(net->hidden_units, k)) + (MOMENTUM * CUIO_FLOATS_ITEM(net->hidden_prev_weights, k * (net->output_n + 1) + j)));
			CUIO_FLOATS_ITEM(net->hidden_weights, k * (net->output_n + 1) + j) += new_dw;
			CUIO_FLOATS_ITEM(net->hidden_prev_weights, k * (net->output_n + 1) + j) = new_dw;
		}
	}
}

void
bpnn_prepare_delta(BPNN *net)
{
	cuio_memcpy_h2d(&net->hidden_delta);
	cuio_memcpy_h2d(&net->input_prev_weights);
	cuio_memcpy_h2d(&net->input_weights);
}

void
bpnn_finalize(BPNN *net)
{
	cuio_memcpy_d2h(&net->input_weights);
	cuio_free_mem(&net->partial_sum);
}

BPNN *
bpnn_create(long n_in, long n_hidden, long n_out)
{
	BPNN	*net;

	net = (BPNN *)calloc(1, sizeof(BPNN));
	if (net == NULL) {
		fprintf(stderr, "bpnn_create: Couldn't allocate neural network\n");
		return NULL;
	}

	net->input_n = n_in;
	net->hidden_n = n_hidden;
	net->output_n = n_out;

	return net;
}

static void
do_load(BPNN *net)
{
	net->input_units = cuio_load_floats("input_units.mem", net->input_n + 1, CUIO_MODE_READONLY);
	net->target = cuio_load_floats("target.mem", net->output_n + 1, CUIO_MODE_READONLY);
	net->input_weights = cuio_load_floats("input_weights.mem", (net->input_n + 1) * (net->hidden_n + 1), CUIO_MODE_READWRITE);
	net->hidden_weights = cuio_load_floats("hidden_weights.mem", (net->hidden_n + 1) * (net->output_n + 1), CUIO_MODE_READWRITE);
	net->input_prev_weights = cuio_load_floats("input_weights.prev.mem", (net->input_n + 1) * (net->hidden_n + 1), CUIO_MODE_WRITEONLY);
	net->hidden_prev_weights = cuio_load_floats("hidden_weights.prev.mem", (net->hidden_n + 1) * (net->output_n + 1), CUIO_MODE_WRITEONLY);
	net->hidden_units = cuio_alloc_mem((net->hidden_n + 1) * sizeof(float));
	net->output_units = cuio_alloc_mem((net->output_n + 1) * sizeof(float));
	net->hidden_delta = cuio_alloc_mem((net->hidden_n + 1) * sizeof(float));
	net->output_delta = cuio_alloc_mem((net->output_n + 1) * sizeof(float));
}

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	BPNN	*net = (BPNN *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot get a network configurations: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%ld %ld %ld", &net->input_n, &net->hidden_n, &net->output_n) != 3) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

BPNN *
bpnn_load(void)
{
	BPNN	*net;

	net = bpnn_create(0, 0, 0);
	cuio_load_conf(confer_load, net);
	do_load(net);
	return net;
}

void
bpnn_save(BPNN *net)
{
	cuio_free_mem(&net->input_units);
	cuio_free_mem(&net->hidden_units);
	cuio_free_mem(&net->target);
	cuio_unload_floats("input_weights.mem", &net->input_weights);
	cuio_unload_floats("hidden_weights.mem", &net->hidden_weights);
	cuio_unload_floats("input_weights.prev.mem", &net->input_prev_weights);
	cuio_unload_floats("hidden_weights.prev.mem", &net->hidden_prev_weights);
	cuio_free_mem(&net->hidden_delta);
	cuio_free_mem(&net->output_delta);
}

void
bpnn_free(BPNN *net)
{
	free((char *)net);
}
