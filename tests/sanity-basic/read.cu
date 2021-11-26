#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "libuxu.h"

#include "cuhelper.h"
#include "timer.h"

__device__ bool	d_result = true;

__global__ void
kernel(uint32_t* g_buf, int seed) 
{
	size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;

	if (g_buf[idx] != (uint32_t)(idx * seed))
		d_result = false;
}

int
main(int argc, char *argv[])
{
	uint32_t	*g_buf;
	size_t	num_tblocks;          
	size_t	num_threads;          
	int	size_order;
	size_t	total_size;
	cudaEvent_t	start_event, stop_event;
	float	kernel_time;
	unsigned free_time;
	unsigned map_time;
	bool h_result = true;
	int seed;

	if (argc != 5) {
		fprintf(stderr, "Usage: %s file size_in_KB threads_per_block seed\n", argv[0]);
		return EXIT_SUCCESS;
	}

	size_order = atoi(argv[2]);
	num_threads = atoi(argv[3]);
	seed = atoi(argv[4]);
    
	//total_size = ((size_t)1 << 30) * size_order;
	total_size = (size_t)1024 * (size_t)size_order;
	num_tblocks = total_size / sizeof(uint32_t) / num_threads;

	CUDA_CALL_SAFE(cudaEventCreate(&start_event));
	CUDA_CALL_SAFE(cudaEventCreate(&stop_event));

	init_tickcount();
	if (uxu_map(argv[1], total_size, UXU_FLAGS_READ, (void **)&g_buf) != UXU_OK)
		return EXIT_FAILURE;
	map_time = get_tickcount();

	CUDA_CALL_SAFE(cudaEventRecord(start_event));
	kernel<<<num_tblocks, num_threads>>>(g_buf, seed);
	CUDA_CALL_SAFE(cudaEventRecord(stop_event));
	
	CUDA_CALL_SAFE(cudaEventSynchronize(stop_event));
	CUDA_CALL_SAFE(cudaEventElapsedTime(&kernel_time, start_event, stop_event));
	
	CUDA_CALL_SAFE(cudaDeviceSynchronize());
	
	init_tickcount();
	if (uxu_unmap(g_buf) != UXU_OK)
		return EXIT_FAILURE;
	free_time = get_tickcount();

	CUDA_CALL_SAFE(cudaMemcpyFromSymbol(&h_result, d_result, sizeof(d_result), 0, cudaMemcpyDeviceToHost));

	printf("==> header: kernel_time (ms),free_time (ms),map_time (ms)\n");
	printf("==> data: %f,%f,%f\n", kernel_time, free_time / 1000.0, map_time / 1000.0);
	printf("==> Data validation: %s\n", h_result ? "Pass" : "Fail");

	if (h_result)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}
