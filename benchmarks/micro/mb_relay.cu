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
"mb_advise <options>\n"
"<options>:\n"
"  -b <# of TB>: number of TB\n"
"  -t <# of threads>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread\n"
"  -S <IO stride>: memory access stride per thread(default: 4k)\n"
"  -r <relay count>: relay count(default: 1)\n"
"  -p <# of partitions>\n"
"  -h: help\n");
}

static CUcontext	context;
static unsigned char	*mem;

static unsigned n_threads = 1;
static unsigned iosize = 1;
static unsigned iostride = 4096;
static unsigned n_tbs = 1;
static unsigned n_relays = 1;
static unsigned	n_parts = 1;
static int	quiet;

static __global__ void
incr(unsigned char *mem, unsigned iosize, unsigned iostride)
{
	unsigned long	idx;
	unsigned	i;

	idx = (blockIdx.x * blockDim.x + threadIdx.x) * iostride;

	if (iosize < sizeof(int)) {
		for (i = 0; i < iosize; i++)
			mem[idx + i] = mem[idx + i] + 1;
	}
	else {
		idx /= sizeof(int);
		for (i = 0; i < iosize / sizeof(int); i++)
			((int *)mem)[idx + i] = ((int *)mem)[idx + i] + 1;
	}
}

static void
incrementing_by_gpu(cudaStream_t strm, unsigned char *mem_my, unsigned char n_tbs_my)
{
	incr<<<n_tbs_my, n_threads, 0, strm>>>(mem_my, iosize, iostride);
	cudaStreamSynchronize(strm);
}

static void
incrementing_by_cpu(unsigned char *mem_my, unsigned char n_tbs_my)
{
	unsigned	i, j, k;

	for (i = 0; i < n_tbs_my; i++) {
		for (j = 0; j < n_threads; j++) {
			unsigned long	idx = (unsigned long)(i * n_threads + j) * iostride;
			if (iosize < sizeof(int)) {
				for (k = 0; k < iosize; k++)
					mem_my[idx + k] = mem_my[idx + k] + 1;
			}
			else {
				idx /= sizeof(int);
				for (k = 0; k < iosize / sizeof(int); k++)
					((int *)mem_my)[idx + k] = ((int *)mem_my)[idx + k] + 1;
			}
		}
	}
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
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "b:t:s:S:Mr:p:hq")) != -1) {
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
		case 'r':
			n_relays = mb_parse_count(optarg, "n_relays");
			break;
		case 'p':
			n_parts = mb_parse_count(optarg, "n_parts");
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
	if (n_parts > n_tbs ) {
		ERROR("# of partition is larger than # of TBs");
	}
}

static void *
do_increment_func(void *arg)
{
	cudaStream_t	strm;
	unsigned	n_part_idx = (unsigned)(unsigned long)arg;
	unsigned char	*mem_my;
	unsigned	n_tbs_my;
	unsigned	i;

	cuCtxSetCurrent(context);
	CUDA_CHECK(cudaStreamCreate(&strm), "cudaStreamCreate");

	n_tbs_my = n_tbs / n_parts;
	mem_my = mem + n_tbs_my * n_threads * iostride * n_part_idx;
	for (i = 0; i < n_relays; i++) {
		incrementing_by_gpu(strm, mem_my, n_tbs_my);
		incrementing_by_cpu(mem_my, n_tbs_my);
	}

	cudaStreamDestroy(strm);

	return NULL;
}

static pthread_t
do_increment(unsigned n_part_idx)
{
	pthread_t	thread;

	if (pthread_create(&thread, NULL, do_increment_func, (void *)(unsigned long)n_part_idx) < 0) {
		ERROR("failed to create thread");
	}
	return thread;
}

static void
init_CU(void)
{
	CUresult	res;
	CUdevice	dev;

	cuInit(0);
	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		ERROR("failed to get device");
        }

	res = cuDevicePrimaryCtxRetain(&context, dev);
	if (res != CUDA_SUCCESS) {
		ERROR("failed to get context");
	}

	res = cuCtxSetCurrent(context);
	if (res != CUDA_SUCCESS) {
		ERROR("failed to set context");
        }
}

int
main(int argc, char *argv[])
{
	pthread_t	*threads;
	unsigned long	size;
	unsigned	ticks, i;

	parse_args(argc, argv);

	init_CU();

	if (!quiet) {
		char	*str_iosize = mb_get_sizestr(iosize);
		char	*str_iostride = mb_get_sizestr(iostride);

		printf("# of tbs: %d, # of threads: %d, IO size: %s, stride: %s\n", n_tbs, n_threads, str_iosize, str_iostride);
		free(str_iosize);
		free(str_iostride);
	}

	size = n_tbs * n_threads * iostride;
	CUDA_CHECK(cudaMallocManaged(&mem, size), "cudaMallocManaged");

	zeroing_by_cpu(mem);

	threads = (pthread_t *)malloc(sizeof(pthread_t) * n_parts);

	init_tickcount();

	for (i = 0; i < n_parts; i++) {
		threads[i] = do_increment(i);
	}

	for (i = 0; i < n_parts; i++) {
		pthread_join(threads[i], NULL);
	}
	ticks = get_tickcount();

	cudaFree(mem);
	free(threads);

	printf("elapsed: %.3f\n", ticks / 1000.0);
	return 0;
}
