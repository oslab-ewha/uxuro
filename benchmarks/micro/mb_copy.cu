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
"  -m <mode>: copy mode: h2g, g2h, g2g, p2p, h2gfp (default: h2g)\n"
"   - h2gfp: host to gpu from peer\n"
"  -t <host memory type>: paged, locked, uvm, uvm_r (default: paged)\n"
"  -k <stride>: stride size for GPU access (default: 64k)\n"
"  -b <TB size>: TB size for GPU access (default: 1)\n"
"  -r <thread size>: thread size for GPU access (default: 16)\n"
"  -h: help\n");
}

typedef enum {
	HOST_TO_GPU, GPU_TO_HOST, GPU_TO_GPU, PEER_TO_PEER, HOST_TO_GPU_FROM_PEER
} mode_copy_t;

typedef enum {
	HMALLOC_PAGED, HMALLOC_LOCKED, HMALLOC_UVM, HMALLOC_UVM_R
} mode_hmalloc_t;

static mode_copy_t	mode_copy = HOST_TO_GPU;
static mode_hmalloc_t	mode_hmalloc = HMALLOC_PAGED;

static unsigned char	*mem_gpu1;
static unsigned char	*mem_gpu2;
static unsigned char	*mem_host;
static unsigned char	*mem_uvm;
static unsigned		*mem_uvm_dummy;
static unsigned	copy_size = (1024 * 1024 * 2);
static unsigned	stride_gpu = (64 * 1024);
static unsigned	n_tb = 1;
static unsigned	n_threads = 16;

static int	quiet;
static int	dummy_sum_cpu;

static void
touching_by_cpu(unsigned char *mem)
{
	unsigned	i;

	for (i = 0; i < copy_size; i += 4096) {
		*(unsigned char *)(mem + i) = 0;
	}
}

static void
reading_by_cpu(unsigned char *mem)
{
	unsigned	i;

	for (i = 0; i < copy_size; i += 4096) {
		dummy_sum_cpu += *(unsigned char *)(mem + i);
	}
}

static void
copying_by_cpu(unsigned char *mem)
{
	if (mode_hmalloc == HMALLOC_UVM_R)
		reading_by_cpu(mem);
	else
		touching_by_cpu(mem);
}

static __global__ void
touching(unsigned char *mem, unsigned copy_size, unsigned stride_gpu)
{
	unsigned	idx;
	unsigned	i;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = idx * stride_gpu; i < copy_size; i += blockDim.x * stride_gpu)
		mem[idx + i] = 0;
}

static void
touching_by_gpu(unsigned char *mem)
{
	touching<<<n_tb, n_threads>>>(mem, copy_size, stride_gpu);
	cudaDeviceSynchronize();
}

static __global__ void
reading(unsigned char *mem, unsigned copy_size, unsigned stride_gpu, unsigned *mem_uvm_dummy)
{
	unsigned	idx;
	unsigned	dummy = 0, i;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = idx * stride_gpu; i < copy_size; i += blockDim.x * stride_gpu)
		dummy += mem[idx + i];
	*mem_uvm_dummy += dummy;
}

static void
reading_by_gpu(unsigned char *mem)
{
	reading<<<n_tb, n_threads>>>(mem, copy_size, stride_gpu, mem_uvm_dummy);
	cudaDeviceSynchronize();
}

static void
copying_by_gpu(unsigned char *mem)
{
	if (mode_hmalloc == HMALLOC_UVM_R)
		reading_by_gpu(mem);
	else
		touching_by_gpu(mem);
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
	else if (strcmp(arg, "h2gfp") == 0)
		return HOST_TO_GPU_FROM_PEER;

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
	else if (strcmp(arg, "uvm_r") == 0)
		return HMALLOC_UVM_R;

	ERROR("invalid host memory allocation mode: %s", arg);

	return HMALLOC_PAGED;
}

static void
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "s:m:t:k:b:r:hq")) != -1) {
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
		case 'k':
			stride_gpu = mb_parse_size(optarg, "stride size");
			break;
		case 'b':
			n_tb = mb_parse_count(optarg, "thread block size");
			break;
		case 'r':
			n_threads = mb_parse_count(optarg, "thread size");
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

	copying_by_cpu(mem_host);

	init_tickcount();

	cudaMemcpy(mem_gpu1, mem_host, copy_size, cudaMemcpyHostToDevice);

	return get_tickcount();
}

static unsigned
copy_host_to_gpu_by_uvm(void)
{
	CUDA_CHECK(cudaMallocManaged(&mem_uvm, copy_size), "cudaMallocManaged");

	copying_by_cpu(mem_uvm);

	if (mode_copy == HOST_TO_GPU_FROM_PEER) {
		CUDA_CHECK(cudaMemAdvise(mem_uvm, copy_size, cudaMemAdviseSetReadMostly, 0), "cudaMemAdvise");
		copying_by_gpu(mem_uvm);
		CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0), "cudaDeviceEnablePeerAccess1");
		copying_by_cpu(mem_uvm);
		CUDA_CHECK(cudaSetDevice(1), "cudaSetDevice");
	}

	init_tickcount();

	copying_by_gpu(mem_uvm);

	return get_tickcount();
}

static unsigned
copy_host_to_gpu(void)
{
	if (mode_hmalloc == HMALLOC_UVM || mode_hmalloc == HMALLOC_UVM_R)
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

	copying_by_gpu(mem_uvm);

	init_tickcount();

	copying_by_cpu(mem_uvm);

	return get_tickcount();
}

static unsigned
copy_gpu_to_host(void)
{
	if (mode_hmalloc == HMALLOC_UVM || mode_hmalloc == HMALLOC_UVM_R)
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

	copying_by_gpu(mem_uvm);

	CUDA_CHECK(cudaSetDevice(1), "cudaSetDevice");
	if (mode_copy == PEER_TO_PEER)
		CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0), "cudaDeviceEnablePeerAccess0");

	init_tickcount();

	copying_by_gpu(mem_uvm);

	return get_tickcount();
}

static unsigned
copy_gpu_to_gpu(void)
{
	if (mode_hmalloc == HMALLOC_UVM || mode_hmalloc == HMALLOC_UVM_R)
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

	if (mode_hmalloc == HMALLOC_UVM_R)
		CUDA_CHECK(cudaMallocManaged(&mem_uvm_dummy, sizeof(int)), "cudaMallocManaged");

	switch (mode_copy) {
	case HOST_TO_GPU:
	case HOST_TO_GPU_FROM_PEER:
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
