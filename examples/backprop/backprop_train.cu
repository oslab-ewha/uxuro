#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

// includes, kernels
#include "backprop_cuda_kernel.cuh"
#include "backprop.h"
#include "timer.h"

#if defined(CUDAMEMCPY)
extern "C" double d2h_memcpy_time;       // in ms
extern "C" double h2d_memcpy_time;       // in ms
#endif

static unsigned
backprop_train_cuda(BPNN *net, float *eo, float *eh)
{
	unsigned long	num_blocks;

	num_blocks = net->input_n / 16;  

	dim3  grid(num_blocks, 1);
	dim3  threads(16, 16);

	unsigned	kernel_ticks = 0;

	bpnn_prepare(net, num_blocks);

	printf("Performing GPU computation\n");

	init_tickcount();

	int	shmsize = (16 + 16 * 16) * sizeof(float);
	bpnn_layerforward_CUDA<<<grid, threads, shmsize>>>(net->kernel_input_units, net->kernel_input_weights, net->kernel_partial_sum,
							   net->input_n, net->hidden_n);
 
	CUDA_CALL_SAFE(cudaDeviceSynchronize());

	kernel_ticks = get_tickcount();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	bpnn_update_hidden(net, num_blocks);
	bpnn_layerforward(net);
	bpnn_output_error(net);
	bpnn_hidden_error(net);
	bpnn_adjust_weights(net);

	bpnn_prepare_delta(net);

	init_tickcount();
	bpnn_adjust_weights_cuda<<<grid, threads>>>(net->kernel_hidden_delta, net->hidden_n, net->kernel_input_units,
						    net->input_n, net->kernel_input_weights, net->kernel_prev_weights);

	CUDA_CALL_SAFE(cudaDeviceSynchronize());
	kernel_ticks += get_tickcount();

	bpnn_finalize(net);

	return kernel_ticks;
}

extern "C" void
backprop_train(void)
{
	BPNN	*net;
	float	out_err, hid_err;
	unsigned	kernel_ticks, pre_ticks, post_ticks;

	init_tickcount();

	net = bpnn_create(layer_size, 16, 1);

	pre_ticks = get_tickcount();

	printf("Input layer size : %ld\n", layer_size);
	printf("Starting training kernel\n");

	kernel_ticks = backprop_train_cuda(net, &out_err, &hid_err);

	init_tickcount();
	bpnn_free(net);
	post_ticks = get_tickcount();

	printf("Training done\n");

	printf("kernel time(us): %u\n", kernel_ticks);
	printf("pre time(us): %u\n", pre_ticks);
	printf("post time(us): %u\n", post_ticks);
}
