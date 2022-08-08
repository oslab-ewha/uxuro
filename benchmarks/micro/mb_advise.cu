#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "mb_common.h"

static void
usage(void)
{
	printf(
"mb_advise <options>\n"
"<options>:\n"
"  -b <# of TB>: number of TB\n"
"  -t <# of threads>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread\n"
"  -S <IO stride>: memory access stride per thread(default: 4k)\n"
"  -l <loop count>: tail GPU loop(default: 1)\n"
"  -M: summing instead of zeroing(do read operation)\n"
"  -a <device no>: cudaMemAdviseSetAccessedBy(cpu or 0, 1, ...)\n"
"  -p <sched type>: cudaMemAdviseSetPreferredLocation(cpu or 0, 1, ...)\n"
"  -R: cudaMemAdviseSetReadMostly\n"
"  -h: help\n");
}

static unsigned n_threads = 1;
static unsigned iosize = 1;
static unsigned iostride = 4096;
static unsigned n_tbs = 1;
static unsigned n_loops_tail = 1;
static int	do_summing = 0;
static int	accessedBy = -1;
static int	preferredLoc = -1;
static int	readMostly;
static int	quiet;

static __device__ unsigned	sum_by_gpu;

/* The summed value has no meaning, which is just for disabling optimization. */
static unsigned	sum_by_cpu;

static __global__ void
zeroing(unsigned char *mem, unsigned iosize, unsigned iostride)
{
	unsigned long	idx;
	unsigned	i;

	idx = (blockIdx.x * blockDim.x + threadIdx.x) * iostride;

	if (iosize < sizeof(int)) {
		for (i = 0; i < iosize; i++)
			mem[idx + i] = 0;
	}
	else {
		idx /= sizeof(int);
		for (i = 0; i < iosize / sizeof(int); i++)
			((int *)mem)[idx + i] = 0;
	}
}

static __global__ void
summing(unsigned char *mem, unsigned iosize, unsigned iostride)
{
	unsigned long	idx;
	unsigned	sum = 0;
	unsigned	i;

	idx = (blockIdx.x * blockDim.x + threadIdx.x) * iostride;

	if (iosize < sizeof(int)) {
		for (i = 0; i < iosize; i++)
			sum += mem[idx + i];
	}
	else {
		idx /= sizeof(int);
		for (i = 0; i < iosize / sizeof(int); i++)
			sum += ((int *)mem)[idx + i];
	}
	sum_by_gpu = sum;
}

static void
zeroing_by_gpu(unsigned char *mem)
{
	zeroing<<<n_tbs, n_threads>>>(mem, iosize, iostride);
	cudaDeviceSynchronize();
}

static void
summing_by_gpu(unsigned char *mem)
{
	summing<<<n_tbs, n_threads>>>(mem, iosize, iostride);
	cudaDeviceSynchronize();
}

static void
zeroing_by_cpu(unsigned char *mem)
{
	unsigned	i, j, k;

	for (i = 0; i < n_tbs; i++) {
		for (j = 0; j < n_threads; j++) {
			unsigned long	idx = (unsigned long)(i * n_threads + j) * iostride;
			if (iosize < sizeof(int)) {
				for (k = 0; k < iosize; k++)
					mem[idx + k] = 0;
			}
			else {
				idx /= sizeof(int);
				for (k = 0; k < iosize / sizeof(int); k++)
					((int *)mem)[idx + k] = 0;
			}
		}
	}
}

static void
summing_by_cpu(unsigned char *mem)
{
	unsigned	i, j, k;

	for (i = 0; i < n_tbs; i++) {
		for (j = 0; j < n_threads; j++) {
			unsigned long	idx = (unsigned long)(i * n_threads + j) * iostride;
			if (iosize < sizeof(int)) {
				for (k = 0; k < iosize; k++)
					sum_by_cpu += mem[idx + k];
			}
			else {
				idx /= sizeof(int);
				for (k = 0; k < iosize / sizeof(int); k++)
					sum_by_cpu += ((int *)mem)[idx + k];
			}
		}
	}
}

static void
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "b:t:s:S:Ma:p:Rl:hq")) != -1) {
		switch (c) {
		case 'b':
			n_tbs = mb_parse_count(optarg, "TB");
			break;
		case 't':
			n_threads = mb_parse_count(optarg, "n_threads");
			break;
		case 's':
			iosize = mb_parse_size(optarg, "IO size");
			break;
		case 'S':
			iostride = mb_parse_size(optarg, "IO stride");
			break;
		case 'M':
			do_summing = 1;
			break;
		case 'a':
			accessedBy = mb_parse_procid(optarg);
			break;
		case 'p':
			preferredLoc = mb_parse_procid(optarg);
			break;
		case 'R':
			readMostly = 1;
			break;
		case 'l':
			n_loops_tail = mb_parse_count(optarg, "n_tail_loops");
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
	if (iostride < iosize) {
		ERROR("IO stride should be larger than IO size");
	}
}

int
main(int argc, char *argv[])
{
	unsigned char	*mem;
	unsigned long	size;
	unsigned	ticks, i;

	parse_args(argc, argv);

	if (!quiet) {
		char	*str_iosize = mb_get_sizestr(iosize);
		char	*str_iostride = mb_get_sizestr(iostride);

		printf("# of tbs: %d, # of threads: %d, IO size: %s, stride: %s\n", n_tbs, n_threads, str_iosize, str_iostride);
		free(str_iosize);
		free(str_iostride);
	}

	size = (unsigned long)n_tbs * n_threads * iostride;
	if (!quiet) {
		char	*str_memsize = mb_get_sizestr(size);
		printf("Managed memory used: %s\n", str_memsize);
		free(str_memsize);
	}
	CUDA_CHECK(cudaMallocManaged(&mem, size), "cudaMallocManaged");
	if (accessedBy >= 0)
		CUDA_CHECK(cudaMemAdvise(mem, size, cudaMemAdviseSetAccessedBy, accessedBy), "cudaMemAdvise/AccessedBy");  // set direct access hint
	if (preferredLoc >= 0)
		CUDA_CHECK(cudaMemAdvise(mem, size, cudaMemAdviseSetPreferredLocation, preferredLoc), "cudaMemAdvise/PreferredLocation");
	if (readMostly)
		CUDA_CHECK(cudaMemAdvise(mem, size, cudaMemAdviseSetReadMostly, 0), "cudaMemAdvise/ReadMostly");

	zeroing_by_gpu(mem);

	init_tickcount();
	for (i = 0; i < n_loops_tail; i++) {
		if (!do_summing) {
			zeroing_by_cpu(mem);
			zeroing_by_gpu(mem);
		}
		else {
			summing_by_cpu(mem);
			summing_by_gpu(mem);
		}
	}
	ticks = get_tickcount();

	{
		unsigned	data;
		CUDA_CHECK(cudaMemcpyFromSymbol(&data, sum_by_gpu, sizeof(unsigned)), "cudaMemcpyFromSymbol");
	}
	cudaFree(mem);

	printf("elapsed: %.3f\n", ticks / 1000.0);
	return 0;
}
