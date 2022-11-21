#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>

#define CUDA_API_PER_THREAD_DEFAULT_STEAM // to Overlap Data Transfers in CUDA Stream
#include "mb_common.h"
#include "timer.h"

#undef PARANOIA // for print vector_add results

static void
usage(void)
{
	printf(
"mb_vectoradd <options>\n"
"<options>:\n"
"  -b <blocks_per_grid>: number of TB per Grid\n"
"  -t <threads_per_block>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread(default: sizeof(int))\n"
"  -p <# of partition>: number of memory partitions\n"
"  -u: allocate memory with uvm (cudaMallocManaged)\n"
"  -q: quiet\n"
"  -h: help\n");
}

static unsigned threads_per_block = 1;
static unsigned blocks_per_grid = 1;
static unsigned	io_size_per_thread = sizeof(int);
static unsigned	partitions = 0;
static int	need_uvm = 0;
static int	quiet;

// Device code
__global__ void
vector_add(int *a, int *b, int *c, unsigned io_count_per_thread)
{
	int	idx = (blockDim.x * blockIdx.x + threadIdx.x) * io_count_per_thread;

	for (int i = 0; i < io_count_per_thread; i++)
		c[idx + i] = a[idx + i] + b[idx + i];
}

static void
read_value_from_cpu(int* mem, unsigned length) {
    int value;

    for (unsigned i = 0; i < length; i++) {
        value = mem[i];
    }
}

// parse user input
static void
parse_args(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "b:t:s:p:uhq")) != -1) {
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
    else if (io_size_per_thread < sizeof(int)) {
        ERROR("IO size should be larger than sizeof(int)");
    }
}

int
main(int argc, char *argv[])
{
	int	*a, *b, *c;
	int	*d_a, *d_b, *d_c;
	unsigned	ticks, i;
	unsigned	n_threads, io_count_per_thread;
	size_t  total_io_size;
	cudaStream_t    *streams;

	parse_args(argc, argv);
	if (!quiet) {
		char	*str_io_size_per_thread = mb_get_sizestr(io_size_per_thread);

		printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %s, partitions: %d\n", threads_per_block, blocks_per_grid, str_io_size_per_thread, partitions);
		free(str_io_size_per_thread);
	}

	n_threads = (unsigned)threads_per_block * blocks_per_grid;
	total_io_size = (size_t)n_threads * io_size_per_thread;
	io_count_per_thread = (unsigned)io_size_per_thread / sizeof(int);
    if (!quiet) {
        char	*str_memsize = mb_get_sizestr(total_io_size);
        printf("Managed memory used: %s\n", str_memsize);
        free(str_memsize);
    }

	if (need_uvm) {
		CUDA_CHECK(cudaMallocManaged((void **)&a, total_io_size), "cudaMallocManaged a");
		CUDA_CHECK(cudaMallocManaged((void **)&b, total_io_size), "cudaMallocManaged b");
		CUDA_CHECK(cudaMallocManaged((void **)&c, total_io_size), "cudaMallocManaged c");
	}
	else {
		CUDA_CHECK(cudaMalloc((void **)&d_a, total_io_size), "cudaMalloc a");
		CUDA_CHECK(cudaMalloc((void **)&d_b, total_io_size), "cudaMalloc b");
		CUDA_CHECK(cudaMalloc((void **)&d_c, total_io_size), "cudaMalloc c");

		if (partitions) {
			CUDA_CHECK(cudaMallocHost(&a, total_io_size), "cudaMallocHost a");
			CUDA_CHECK(cudaMallocHost(&b, total_io_size), "cudaMallocHost b");
			CUDA_CHECK(cudaMallocHost(&c, total_io_size), "cudaMallocHost c");
		}
		else {
			a = (int *)malloc(total_io_size);
			b = (int *)malloc(total_io_size);
			c = (int *)malloc(total_io_size);
		}
	}

	init_tickcount();

	for (i = 0; i < n_threads * io_count_per_thread; i++) {
		a[i] = i % 1024;    // to avoid exceeding the integer range, limit the element value of 'vector a' to between 0 and 1023.
		b[i] = 1;
	}

	if (need_uvm) {
		vector_add<<<blocks_per_grid, threads_per_block>>>(a, b, c, io_count_per_thread);
		cudaDeviceSynchronize();
	}
	else if (partitions) {
		unsigned    n_threads_part = n_threads / partitions;
		unsigned	io_size_part = (unsigned)total_io_size / partitions;
		unsigned	offset = 0;
		streams = (cudaStream_t *)malloc(partitions * sizeof(cudaStream_t));

		for (i = 0; i < partitions; i++) {
			CUDA_CHECK(cudaStreamCreate(&streams[i]), "cudaStreamCreate");

			CUDA_CHECK(cudaMemcpyAsync(&d_a[offset], &a[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpyAsync a");
			CUDA_CHECK(cudaMemcpyAsync(&d_b[offset], &b[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpyAsync b");
            vector_add<<<blocks_per_grid / partitions, threads_per_block, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, io_count_per_thread);
			CUDA_CHECK(cudaMemcpyAsync(&c[offset], &d_c[offset], io_size_part, cudaMemcpyDeviceToHost, streams[i]), "cudaMemcpyAsync c");

			offset += n_threads_part * io_count_per_thread;
		}

		for (i = 0; i < partitions; i++)
			cudaStreamSynchronize(streams[i]);
	}
	else {
		CUDA_CHECK(cudaMemcpy(d_a, a, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy a");
		CUDA_CHECK(cudaMemcpy(d_b, b, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy b");
        vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, io_count_per_thread);
		CUDA_CHECK(cudaMemcpy(c, d_c, total_io_size, cudaMemcpyDeviceToHost), "cudaMemcpy c");
		cudaDeviceSynchronize();
	}

    read_value_from_cpu(c, n_threads * io_count_per_thread);

	ticks = get_tickcount();

// print result
#ifdef PARANOIA
for (i = 0; i < n_threads * io_count_per_thread; i++) {
	if (i % threads_per_block == 0)
		printf("\n");
	printf("%d:%d/ ", i, c[i]);
}
printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %s, partitions: %d\n", threads_per_block, blocks_per_grid, io_size_per_thread, partitions);
#endif

	if (need_uvm) {
		cudaFree(a);
		cudaFree(b);
		cudaFree(c);
	}
	else {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

		if (partitions) {
			cudaFreeHost(a);
			cudaFreeHost(b);
			cudaFreeHost(c);
		}
		else {
			free(a);
			free(b);
			free(c);
		}
	}

	if (partitions) {
		for (i = 0; i < partitions; i++)
			cudaStreamDestroy(streams[i]);
	}

	printf("elapsed: %.3f\n", ticks / 1000.0);
	return 0;
}