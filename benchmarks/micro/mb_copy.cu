#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#include <cuda.h>

#include "mb_common.h"

static void
usage(void)
{
	printf(
"mb_copy <options>\n"
"<options>:\n"
"  -s <copy size>: read/write size in byte per thread (default: 2MB)\n"
"  -m <mode>: copy mode: h2g, g2h, g2g, p2p (default: h2g)\n"
"  -t <host memory type>: paged, locked, uvm (default: paged)\n"
"  -h: help\n");
}

typedef enum {
	HOST_TO_GPU, GPU_TO_HOST, GPU_TO_GPU, PEER_TO_PEER 
} mode_copy_t;

typedef enum {
	HMALLOC_PAGED, HMALLOC_LOCKED, HMALLOC_UVM
} mode_hmalloc_t;

static mode_copy_t	mode_copy = HOST_TO_GPU;
static mode_hmalloc_t	mode_hmalloc = HMALLOC_PAGED;

static unsigned char	*mem_gpu1;
static unsigned char	*mem_gpu2;
static unsigned char	*mem_host;
static unsigned char	*mem_uvm;
static unsigned	copy_size = (1024 * 1024 * 2);
static int	quiet;

static void
touching_by_cpu(unsigned char *mem)
{
	unsigned	i;

	for (i = 0; i < copy_size; i += 4096) {
		*(unsigned char *)(mem + i) = 0;
	}
}

#define TWO_MB	(1024 * 1024 * 2)

static __global__ void
touching(unsigned char *mem, unsigned copy_size)
{
	unsigned	idx;
	unsigned	i;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = idx * TWO_MB; i < copy_size; i += blockDim.x * TWO_MB)
		mem[idx + i] = 0;
}

static void
touching_by_gpu(unsigned char *mem)
{
	touching<<<1, 16>>>(mem, copy_size);
	cudaDeviceSynchronize();
}

static mode_copy_t
parse_mode_copy(const char *arg)
{
	if (strcmp(arg, "h2g") == 0)
		return HOST_TO_GPU;
	else if (strcmp(arg, "g2h") == 0)
		return GPU_TO_HOST;
	else if (strcmp(arg, "g2g") == 0)
		return GPU_TO_GPU;
	else if (strcmp(arg, "p2p") == 0)
		return PEER_TO_PEER;

	ERROR("invalid copy mode: %s", arg);

	return HOST_TO_GPU;
}

static mode_hmalloc_t
parse_mode_hmalloc(const char *arg)
{
	if (strcmp(arg, "paged") == 0)
		return HMALLOC_PAGED;
	else if (strcmp(arg, "locked") == 0)
		return HMALLOC_LOCKED;
	else if (strcmp(arg, "uvm") == 0)
		return HMALLOC_UVM;

	ERROR("invalid host memory allocation mode: %s", arg);

	return HMALLOC_PAGED;
}

static void
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "s:m:t:hq")) != -1) {
		switch (c) {
		case 's':
			copy_size = mb_parse_size(optarg, "copy size");
			break;
		case 'm':
			mode_copy = parse_mode_copy(optarg);
			break;
		case 't':
			mode_hmalloc = parse_mode_hmalloc(optarg);
			break;
		case 'q':
			quiet = 1;
			break;
		case 'h':
			usage();
			exit(0);
		default:
			usage();
			ERROR("invalid argument");
		}
	}
}

static unsigned char *
alloc_mem_host(unsigned size)
{
	unsigned char	*mem;

	if (mode_hmalloc == HMALLOC_PAGED) {
		mem = (unsigned char *)malloc(size);
		if (mem == NULL)
			ERROR("cannot allocate host memory");
	}
	else {
		CUDA_CHECK(cudaMallocHost(&mem, size), "cudaMallocHost");
	}
	return mem;
}

static unsigned
copy_host_to_gpu_by_memcpy(void)
{
	mem_host = alloc_mem_host(copy_size);

	CUDA_CHECK(cudaMalloc(&mem_gpu1, copy_size), "cudaMalloc on GPU");

	touching_by_cpu(mem_host);

	init_tickcount();

	cudaMemcpy(mem_gpu1, mem_host, copy_size, cudaMemcpyHostToDevice);

	return get_tickcount();
}

static unsigned
copy_host_to_gpu_by_uvm(void)
{
	CUDA_CHECK(cudaMallocManaged(&mem_uvm, copy_size), "cudaMallocManaged");

	touching_by_cpu(mem_uvm);

	init_tickcount();

	touching_by_gpu(mem_uvm);

	return get_tickcount();
}

static unsigned
copy_host_to_gpu(void)
{
	if (mode_hmalloc == HMALLOC_UVM)
		return copy_host_to_gpu_by_uvm();
	else
		return copy_host_to_gpu_by_memcpy();
}

static unsigned
copy_gpu_to_host_by_memcpy(void)
{
	mem_host = alloc_mem_host(copy_size);

	CUDA_CHECK(cudaMalloc(&mem_gpu1, copy_size), "cudaMalloc on GPU");

	touching_by_cpu(mem_host);

	init_tickcount();

	cudaMemcpy(mem_host, mem_gpu1, copy_size, cudaMemcpyDeviceToHost);

	return get_tickcount();
}

static unsigned
copy_gpu_to_host_by_uvm(void)
{
	CUDA_CHECK(cudaMallocManaged(&mem_uvm, copy_size), "cudaMallocManaged");

	touching_by_gpu(mem_uvm);

	init_tickcount();

	touching_by_cpu(mem_uvm);

	return get_tickcount();
}

static unsigned
copy_gpu_to_host(void)
{
	if (mode_hmalloc == HMALLOC_UVM)
		return copy_gpu_to_host_by_uvm();
	else
		return copy_gpu_to_host_by_memcpy();
}

static unsigned
copy_gpu_to_gpu_by_memcpy(void)
{
	CUDA_CHECK(cudaSetDevice(0), "cudaSetDevice");
	if (mode_copy == PEER_TO_PEER)
		CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0), "cudaDeviceEnablePeerAccess1");
	CUDA_CHECK(cudaMalloc(&mem_gpu1, copy_size), "cudaMalloc on GPU1");

	CUDA_CHECK(cudaSetDevice(1), "cudaSetDevice");
	if (mode_copy == PEER_TO_PEER)
		CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0), "cudaDeviceEnablePeerAccess0");
	CUDA_CHECK(cudaMalloc(&mem_gpu2, copy_size), "cudaMalloc on GPU2");

	init_tickcount();

	cudaMemcpy(mem_gpu2, mem_gpu1, copy_size, cudaMemcpyDeviceToDevice);

	return get_tickcount();
}

static unsigned
copy_gpu_to_gpu_by_uvm(void)
{
	CUDA_CHECK(cudaSetDevice(0), "cudaSetDevice");
	if (mode_copy == PEER_TO_PEER)
		CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0), "cudaDeviceEnablePeerAccess1");
	CUDA_CHECK(cudaMallocManaged(&mem_uvm, copy_size), "cudaMallocManaged");

	touching_by_gpu(mem_uvm);

	CUDA_CHECK(cudaSetDevice(1), "cudaSetDevice");
	if (mode_copy == PEER_TO_PEER)
		CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0), "cudaDeviceEnablePeerAccess0");

	init_tickcount();

	touching_by_gpu(mem_uvm);

	return get_tickcount();
}

static unsigned
copy_gpu_to_gpu(void)
{
	if (mode_hmalloc == HMALLOC_UVM)
		return copy_gpu_to_gpu_by_uvm();
	else
		return copy_gpu_to_gpu_by_memcpy();
}

int
main(int argc, char *argv[])
{
	unsigned	ticks;

	parse_args(argc, argv);

	if (!quiet) {
		char	*str_size = mb_get_sizestr(copy_size);

		printf("copy size: %s\n", str_size);
		free(str_size);
	}

	switch (mode_copy) {
	case HOST_TO_GPU:
		ticks = copy_host_to_gpu();
		break;
	case GPU_TO_HOST:
		ticks = copy_gpu_to_host();
		break;
	case GPU_TO_GPU:
	case PEER_TO_PEER:
		ticks = copy_gpu_to_gpu();
		break;
	default:
		break;
	}

	printf("elapsed: %.3f\n", ticks / 1000.0);
	return 0;
}
