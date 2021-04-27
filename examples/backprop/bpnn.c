#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "backprop.h"
#include "cudaio.h"

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

extern long	layer_size;

extern void store_data(const char *fname, size_t count, float *data);
extern void free_data(float *data);

static float
drnd(void)
{
	return ((float)rand() / (float)BIGRND);
}

static float
squash(float x)
{
	//x = -x;
	//m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
	//return(1.0 / (1.0 + m));
	return (1.0 / (1.0 + exp(-x)));
}

static void
zero_prev_weights(BPNN *net)
{
	memset(net->input_prev_weights, 0, sizeof(float) * (net->input_n + 1) * (net->hidden_n + 1));
	memset(net->hidden_prev_weights, 0, sizeof(float) * (net->hidden_n + 1) * (net->output_n + 1));
}

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

void
bpnn_prepare(BPNN *net, unsigned long num_blocks)
{
#ifdef CUDAMEMCPY
	CUDA_CALL_SAFE(cudaMalloc((void **)&net->kernel_input_units, (net->input_n + 1) * sizeof(float)));
	CUDA_CALL_SAFE(cudaMalloc((void **)&net->kernel_input_weights, (net->input_n + 1) * (net->hidden_n + 1) * sizeof(float)));

	CUDA_CALL_SAFE(cudaMemcpy(net->kernel_input_units, net->input_units, (net->input_n + 1) * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL_SAFE(cudaMemcpy(net->kernel_input_weights, net->input_weights, (net->input_n + 1) * (net->hidden_n + 1) * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CALL_SAFE(cudaMalloc((void **)&net->kernel_partial_sum, num_blocks * WIDTH * sizeof(float)));

	CUDA_CALL_SAFE(cudaDeviceSynchronize());
#else
	net->kernel_input_units = net->input_units;
	net->kernel_input_weights = net->input_weights;
	CUDA_CALL_SAFE(cudaMallocManaged((void **)&net->partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemAttachGlobal));
	net->kernel_partial_sum = net->partial_sum;
#endif
}

void
bpnn_update_hidden(BPNN *net, unsigned long num_blocks)
{
	long	j;

#if defined(CUDAMEMCPY)
	net->partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));
	CUDA_CALL_SAFE(cudaMemcpy(net->partial_sum, net->kernel_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
#else
	net->partial_sum = net->kernel_partial_sum;
#endif

	for (j = 1; j <= net->hidden_n; j++) {
		double	sum;
		long	k;

		sum = 0.0;
		for (k = 0; k < num_blocks; k++) {
			sum += net->partial_sum[k * net->hidden_n + j - 1];
		}
		sum += net->input_weights[j];
		net->hidden_units[j] = (float)(1.0 / (1.0 + exp(-sum)));
	}
}

void
bpnn_layerforward(BPNN *net)
{
	float	sum;
	long	j, k;

	/*** Set up thresholding unit ***/
	net->hidden_units[0] = 1.0;
	/*** For each unit in second layer ***/
	for (j = 1; j <= net->output_n; j++) {
		/*** Compute weighted sum of its inputs ***/
		sum = 0.0;
		for (k = 0; k <= net->hidden_n; k++) {	
			sum += net->hidden_weights[k * (net->output_n + 1) + j] * net->hidden_units[k]; 
		}
		net->output_units[j] = squash(sum);
	}
}

float
bpnn_output_error(BPNN *net)
{
	long	j;
	float	o, t, errsum;

	errsum = 0.0;
	for (j = 1; j <= net->output_n; j++) {
		o = net->output_units[j];
		t = net->target[j];
		net->output_delta[j] = o * (1.0 - o) * (t - o);
		errsum += ABS(net->output_delta[j]);
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
		h = net->hidden_units[j];
		sum = 0.0;
		for (k = 1; k <= net->output_n; k++) {
			sum += net->output_delta[k] * net->hidden_weights[j * (net->output_n + 1) + k];
		}
		net->hidden_delta[j] = h * (1.0 - h) * sum;
		errsum += ABS(net->hidden_delta[j]);
	}
	return errsum;
}

void
bpnn_adjust_weights(BPNN *net)
{
	float	new_dw;
	long	k, j;

	net->hidden_units[0] = 1.0;
	//eta = 0.3;
	//momentum = 0.3;

	for (j = 1; j <= net->output_n; j++) {
		for (k = 0; k <= net->hidden_n; k++) {
			new_dw = ((ETA * net->output_delta[j] * net->hidden_units[k]) + (MOMENTUM * net->hidden_prev_weights[k * (net->output_n + 1) + j]));
			net->hidden_weights[k * (net->output_n + 1) + j] += new_dw;
			net->hidden_prev_weights[k * (net->output_n + 1) + j] = new_dw;
		}
	}
}

void
bpnn_prepare_delta(BPNN *net)
{
#if defined(CUDAMEMCPY)
	CUDA_CALL_SAFE(cudaMalloc((void**)&net->kernel_hidden_delta, (net->hidden_n + 1) * sizeof(float)));
	CUDA_CALL_SAFE(cudaMalloc((void**)&net->kernel_prev_weights, (net->input_n + 1) * (net->hidden_n + 1) * sizeof(float)));

	CUDA_CALL_SAFE(cudaMemcpy(net->kernel_hidden_delta, net->hidden_delta, (net->hidden_n + 1) * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL_SAFE(cudaMemcpy(net->kernel_prev_weights, net->input_prev_weights, (net->input_n + 1) * (net->hidden_n + 1) * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL_SAFE(cudaMemcpy(net->kernel_input_weights, net->input_weights, (net->input_n + 1) * (net->hidden_n + 1) * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CALL_SAFE(cudaDeviceSynchronize());
#else
	net->kernel_hidden_delta = net->hidden_delta;
	net->kernel_prev_weights = net->input_prev_weights;
	net->kernel_input_weights = net->input_weights;
#endif
}

void
bpnn_finalize(BPNN *net)
{
#if defined(CUDAMEMCPY)
	CUDA_CALL_SAFE(cudaMemcpy(net->input_weights, net->kernel_input_weights, (net->input_n + 1) * (net->hidden_n + 1) * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL_SAFE(cudaFree(net->kernel_input_units));
	CUDA_CALL_SAFE(cudaFree(net->kernel_input_weights));
	CUDA_CALL_SAFE(cudaFree(net->kernel_prev_weights));
	CUDA_CALL_SAFE(cudaFree(net->kernel_hidden_delta));

	free(net->partial_sum);
#endif

	CUDA_CALL_SAFE(cudaFree(net->partial_sum));
}

static void *
load_floats(const char *fname, size_t count, cuio_mode_t mode)
{
	char	fpath[256];

	sprintf(fpath, "%s/%s", folder, fname);
	return cuio_load_floats(fpath, count, mode);
}

static void
unload_floats(const char *fname, size_t count, float *data, cuio_mode_t mode)
{
	char	fpath[256];

	sprintf(fpath, "%s/%s", folder, fname);
	cuio_unload_floats(fpath, count, data, mode);
}

BPNN *
bpnn_create(long n_in, long n_hidden, long n_out)
{
	BPNN	*newnet;

	newnet = (BPNN *)malloc(sizeof(BPNN));
	if (newnet == NULL) {
		fprintf(stderr, "bpnn_create: Couldn't allocate neural network\n");
		return NULL;
	}

	newnet->input_n = n_in;
	newnet->hidden_n = n_hidden;
	newnet->output_n = n_out;

#ifdef GENERATOR
	newnet->input_units = load_floats(NULL, n_in + 1, CUIO_MODE_NONE);
	newnet->target = load_floats(NULL, n_out + 1, CUIO_MODE_NONE);
	newnet->input_weights = load_floats(NULL, (n_in + 1) * (n_hidden + 1), CUIO_MODE_NONE);
	newnet->hidden_weights = load_floats(NULL, (n_hidden + 1) * (n_out + 1), CUIO_MODE_NONE);	
	randomize_bpnn(newnet);
#else
	newnet->input_units = load_floats("input_units.mem", n_in + 1, CUIO_MODE_READONLY);
	newnet->target = load_floats("target.mem", n_out + 1, CUIO_MODE_READONLY);
	newnet->input_weights = load_floats("input_weights.mem", (n_in + 1) * (n_hidden + 1), CUIO_MODE_READWRITE);
	newnet->hidden_weights = load_floats("hidden_weights.mem", (n_hidden + 1) * (n_out + 1), CUIO_MODE_READWRITE);
#endif
	newnet->input_prev_weights = load_floats("input_weights.prev.mem", (n_in + 1) * (n_hidden + 1), CUIO_MODE_WRITEONLY);
	newnet->hidden_prev_weights = load_floats("hidden_weights.prev.mem", (n_hidden + 1) * (n_out + 1), CUIO_MODE_WRITEONLY);
	newnet->hidden_units = load_floats(NULL, n_hidden + 1, CUIO_MODE_NONE);
	newnet->output_units = load_floats(NULL, n_out + 1, CUIO_MODE_NONE);
	newnet->hidden_delta = load_floats(NULL, n_hidden + 1, CUIO_MODE_NONE);
	newnet->output_delta = load_floats(NULL, n_out + 1, CUIO_MODE_NONE);

#ifdef CUDAMEMCPY
	zero_prev_weights(newnet);
#endif	
	return newnet;
}

void
bpnn_free(BPNN *net)
{
#ifdef GENERATOR
	unload_floats("input_units.mem", net->input_n + 1, net->input_units, CUIO_MODE_WRITEONLY);
	unload_floats("target.mem", net->output_n + 1, net->target, CUIO_MODE_WRITEONLY);
	unload_floats("input_weights.mem", (net->input_n + 1) * (net->hidden_n + 1), net->input_weights, CUIO_MODE_WRITEONLY);
	unload_floats("hidden_weights.mem", (net->hidden_n + 1) * (net->output_n + 1), net->hidden_weights, CUIO_MODE_WRITEONLY);
#else
	unload_floats("input_units.mem", net->input_n + 1, net->input_units, CUIO_MODE_READONLY);
	unload_floats(NULL, net->hidden_n + 1, net->hidden_units, CUIO_MODE_NONE);
	unload_floats("target.mem", net->output_n + 1, net->target, CUIO_MODE_READONLY);
	unload_floats("input_weights.mem", (net->input_n + 1) * (net->hidden_n + 1), net->input_weights, CUIO_MODE_READWRITE);
	unload_floats("hidden_weights.mem", (net->hidden_n + 1) * (net->output_n + 1), net->hidden_weights, CUIO_MODE_READWRITE);
	unload_floats("input_weights.prev.mem", (net->input_n + 1) * (net->hidden_n + 1), net->input_prev_weights, CUIO_MODE_WRITEONLY);
	unload_floats("hidden_weights.prev.mem", (net->hidden_n + 1) * (net->output_n + 1), net->hidden_prev_weights, CUIO_MODE_WRITEONLY);
	unload_floats(NULL, net->hidden_n + 1, net->hidden_delta, CUIO_MODE_NONE);
	unload_floats(NULL, net->output_n + 1, net->output_delta, CUIO_MODE_NONE);
#endif
	free((char *)net);
}
