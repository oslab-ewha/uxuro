#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "mb_common.h"
#include "timer.h"

#undef PARANOIA // for print VecAdd results

static void
usage(void)
{
	printf(
"mb_copy <options>\n"
"<options>:\n"
"  -b <blocks_per_grid>: number of TB per Grid\n"
"  -t <threads_per_block>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread(default: sizeof(int))\n"
"  -p <# of partition>: number of memory partitions\n"
"  -u: allocate memory with uvm (cudaMallocManaged)\n"
"  -q: quiet\n"
"  -h: help\n");
}

static unsigned	threads_per_block = 1;
static unsigned	blocks_per_grid = 1;
static unsigned	io_size_per_thread = sizeof(int);
static unsigned	partitions = 0;
static int	need_uvm = 0;
static int	quiet;

// Device code
__global__ void
VecAdd(int *A, int *B, int *C, unsigned io_count_per_thread)
{
	int	idx = (blockDim.x * blockIdx.x + threadIdx.x) * io_count_per_thread;

	for (int i = 0; i < io_count_per_thread; i++)
		C[idx + i] = A[idx + i] + B[idx + i];
}

// parse user input
static void
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "b:t:s:cp:uhq")) != -1) {
		switch (c) {
		case 't':
			threads_per_block = mb_parse_count(optarg, "threads_per_block");
			break;
		case 'b':
			blocks_per_grid = mb_parse_count(optarg, "blocks_per_grid");
			break;
		case 's':
			io_size_per_thread = mb_parse_size(optarg, "IO size");
			break;
		case 'p':
			partitions = mb_parse_count(optarg, "partitions");
			break;
		case 'u':
			need_uvm = 1;
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

	if (partitions && blocks_per_grid % partitions != 0) {
		ERROR("blocks_per_grid should be multiples of partitions");
	}
}

int
main(int argc, char *argv[])
{
	int	*A, *B, *C;
	int	*d_A, *d_B, *d_C;
	unsigned	ticks, i;
	unsigned long	n_threads, total_io_size, io_count_per_thread;
	cudaStream_t	*streams;

	parse_args(argc, argv);
	if (!quiet) {
		char	*str_io_size_per_thread = mb_get_sizestr(io_size_per_thread);

		printf("threads_per_block: %d, blocks_per_grid: %d, IO size: %s, partitions: %d\n", threads_per_block, blocks_per_grid, str_io_size_per_thread, partitions);
		free(str_io_size_per_thread);
	}
	n_threads = (unsigned long)threads_per_block * blocks_per_grid;
	total_io_size = (unsigned long)n_threads * io_size_per_thread;
	io_count_per_thread = (unsigned long)io_size_per_thread / sizeof(int);

	if (need_uvm) {
		cudaMallocManaged(&A, total_io_size);
		cudaMallocManaged(&B, total_io_size);
		cudaMallocManaged(&C, total_io_size);
	}
	else {
		A = (int *)malloc(total_io_size);
		B = (int *)malloc(total_io_size);
		C = (int *)malloc(total_io_size);

		cudaMalloc((void **)&d_A, total_io_size);
		cudaMalloc((void **)&d_B, total_io_size);
		cudaMalloc((void **)&d_C, total_io_size);
	}

	init_tickcount();

	for (i = 0; i < n_threads * io_count_per_thread; i++) {
		A[i] = i + 1;
		B[i] = 1;
	}

	if (need_uvm) {
		VecAdd<<<blocks_per_grid, threads_per_block>>>(A, B, C, io_count_per_thread);
		cudaDeviceSynchronize();
	}
	else if (partitions) {
		unsigned long	n_threads_part = n_threads / partitions;
		unsigned long	io_size_part = total_io_size / partitions;
		unsigned long	offset = 0;
		streams = (cudaStream_t *)malloc(partitions * sizeof(cudaStream_t));

		for (i = 0; i < partitions; i++) {
			cudaStreamCreate(&streams[i]);

			cudaMemcpyAsync(&d_A[offset], &A[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]);
			cudaMemcpyAsync(&d_B[offset], &B[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]);
			VecAdd<<<blocks_per_grid / partitions, threads_per_block, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, io_count_per_thread);
			cudaMemcpyAsync(&C[offset], &d_C[offset], io_size_part, cudaMemcpyDeviceToHost, streams[i]);

			offset += n_threads_part * io_count_per_thread;
		}

		for (i = 0; i < partitions; i++)
			cudaStreamSynchronize(streams[i]);
	}
	else {
		cudaMemcpy(d_A, A, total_io_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, total_io_size, cudaMemcpyHostToDevice);
		VecAdd<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, io_count_per_thread);
		cudaMemcpy(C, d_C, total_io_size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}

	ticks = get_tickcount();

// print result
#ifdef PARANOIA
for (i = 0; i < n_threads * io_count_per_thread; i++) {
	if (i % threads_per_block == 0)
		printf("\n");
	printf("%d:%d/ ", i, C[i]);
}
printf("\nthreads_per_block: %d, blocks_per_grid: %d, N: %lu, total IO size: %lu, IO size: %d\n", threads_per_block, blocks_per_grid, n_threads, total_io_size, io_size_per_thread);
#endif

	if (need_uvm) {
		cudaFree(A);
		cudaFree(B);
		cudaFree(C);
	}
	else {
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		free(A);
		free(B);
		free(C);
	}

	if (partitions) {
		for (i = 0; i < partitions; i++)
			cudaStreamDestroy(streams[i]);
	}

	printf("elapsed: %.3f\n", ticks / 1000.0);
	return 0;
}