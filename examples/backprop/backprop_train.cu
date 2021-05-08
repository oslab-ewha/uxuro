#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "backprop_cuda_kernel.cuh"
#include "backprop.h"
#include "timer.h"

typedef struct {
	unsigned	kernel_ticks_gpu, kernel_ticks_cpu;
	float		err_out, err_hid;
} train_result_t;

static void
backprop_train_cuda(BPNN *net, train_result_t *pres)
{
	unsigned long	num_blocks;
	int	shmsize;

	num_blocks = net->input_n / 16;  

	dim3  grid(num_blocks, 1);
	dim3  threads(16, 16);

	bpnn_prepare(net, num_blocks);

	printf("Performing GPU computation\n");

	shmsize = (16 + 16 * 16) * sizeof(float);

	init_tickcount();
	bpnn_layerforward_CUDA<<<grid, threads, shmsize>>>(CUIO_FLOATS_D(net->input_units),
							   CUIO_FLOATS_D(net->input_weights),
							   CUIO_FLOATS_D(net->partial_sum),
							   net->input_n, net->hidden_n);
 
	CUDA_CALL_SAFE(cudaDeviceSynchronize());

	pres->kernel_ticks_gpu = get_tickcount();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	init_tickcount();

	bpnn_update_hidden(net, num_blocks);
	bpnn_layerforward(net);
	pres->err_out = bpnn_output_error(net);
	pres->err_hid = bpnn_hidden_error(net);
	bpnn_adjust_weights(net);

	bpnn_prepare_delta(net);

	pres->kernel_ticks_cpu = get_tickcount();

	init_tickcount();
	bpnn_adjust_weights_cuda<<<grid, threads>>>(CUIO_FLOATS_D(net->hidden_delta), net->hidden_n,
						    CUIO_FLOATS_D(net->input_units),  net->input_n,
						    CUIO_FLOATS_D(net->input_weights), CUIO_FLOATS_D(net->input_prev_weights));

	CUDA_CALL_SAFE(cudaDeviceSynchronize());
	pres->kernel_ticks_gpu += get_tickcount();

	bpnn_finalize(net);
}

extern "C" void
backprop_train(const char *folder)
{
	BPNN	*net;
	unsigned	pre_ticks, post_ticks;
	train_result_t	res;

	init_tickcount();

	net = bpnn_load(folder);

	pre_ticks = get_tickcount();

	printf("Network: %ldx%ldx%ld\n", net->input_n, net->hidden_n, net->output_n);
	printf("Starting training kernel\n");

	backprop_train_cuda(net, &res);

	init_tickcount();
	bpnn_save(net);
	bpnn_free(net);
	post_ticks = get_tickcount();

	printf("Training done\n");
	printf("Output Error: %f\n", res.err_out);
	printf("Hidden Error: %f\n", res.err_hid);

	printf("pre time(us): %u\n", pre_ticks);
	printf("kernel time(us): %u(gpu:%u)\n", res.kernel_ticks_gpu + res.kernel_ticks_cpu, res.kernel_ticks_gpu);
	printf("post time(us): %u\n", post_ticks);
}
