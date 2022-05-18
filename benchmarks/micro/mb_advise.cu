#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "benchmark.h"
#include "timer.h"

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
static int	accessedBy = -1;
static int	preferredLoc = -1;
static int	readMostly;
static int	quiet;

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

static void
zeroing_by_gpu(unsigned char *mem)
{
	zeroing<<<n_tbs, n_threads>>>(mem, iosize, iostride);
	cudaDeviceSynchronize();
}

static void
zeroing_by_cpu(unsigned char *mem)
{
	unsigned	i, j, k;

	for (i = 0; i < n_tbs; i++)
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

static int
parse_count(const char *arg, const char *name)
{
	if (sscanf(optarg, "%u", &n_threads) != 1)
		ERROR("invalid number of %s: %s", arg, name);
	if (n_threads == 0)
		ERROR("0 %s not allowed", name);
	return n_threads;
}

static int
parse_size(const char *arg, const char *name)
{
	unsigned	size;
	char	unit;

	if (sscanf(optarg, "%u%c", &size, &unit) == 2) {
		if (unit == 'k')
			size *= 1024;
		else if (unit == 'm')
			size *= 1024 * 1024;
		else if (unit == 'g')
			size *= 1024 * 1024 * 1024;
		else
			ERROR("invalid size unit");
	}
	if (size == 0)
		ERROR("0 %s is not allowed", name);
	return size;
}

static int
parse_procid(const char *arg)
{
	unsigned	gpuid;

	if (strcmp(arg, "cpu") == 0)
		return cudaCpuDeviceId;

	if (sscanf(arg, "%u", &gpuid) != 1)
		ERROR("invalid argument: %s\n", arg);
	return gpuid;
}

static void
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "b:t:s:S:a:p:Rl:hq")) != -1) {
		switch (c) {
		case 'b':
			n_tbs = parse_count(optarg, "TB");
			break;
		case 't':
			n_threads = parse_count(optarg, "n_threads");
			break;
		case 's':
			iosize = parse_size(optarg, "IO size");
			break;
		case 'S':
			iostride = parse_size(optarg, "IO stride");
			break;
		case 'a':
			accessedBy = parse_procid(optarg);
			break;
		case 'p':
			preferredLoc = parse_procid(optarg);
			break;
		case 'R':
			readMostly = 1;
			break;
		case 'l':
			n_loops_tail = parse_count(optarg, "n_tail_loops");
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

static char *
get_sizestr(unsigned long num)
{
	char	buf[1024];

	if (num < 1024)
		snprintf(buf, 1024, "%lu", num);
	else {
		num /= 1024;
		if (num < 1024)
			snprintf(buf, 1024, "%luk", num);
		else {
			num /= 1024;
			if (num < 1024)
				snprintf(buf, 1024, "%lum", num);
			else {
				num /= 1024;
				snprintf(buf, 1024, "%lug", num);
			}
		}
	}
	return strdup(buf);
}

int
main(int argc, char *argv[])
{
	unsigned char	*mem;
	unsigned long	size;
	unsigned	ticks, i;

	parse_args(argc, argv);

	if (!quiet) {
		char	*str_iosize = get_sizestr(iosize);
		char	*str_iostride = get_sizestr(iostride);

		printf("# of tbs: %d, # of threads: %d, IO size: %s, stride: %s\n", n_tbs, n_threads, str_iosize, str_iostride);
		free(str_iosize);
		free(str_iostride);
	}

	size = n_tbs * n_threads * iostride;
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
		zeroing_by_cpu(mem);
		zeroing_by_gpu(mem);
	}
	ticks = get_tickcount();

	cudaFree(mem);

	printf("elapsed: %.3f\n", ticks / 1000.0);
	return 0;
}
