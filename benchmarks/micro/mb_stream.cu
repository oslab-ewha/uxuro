/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)
    SCALE:      a(i) = q * b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q * c(i)

  It measures the memory system on the device.
  The implementation is in single precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Original written by: Massimiliano Fatica, NVIDIA Corporation
  Edited by: JeongHa Lee, to compare uvm with naive ver.
*/

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/mman.h>

#define CUDA_API_PER_THREAD_DEFAULT_STEAM // to Overlap Data Transfers in CUDA Stream
#include "mb_common.h"
#include "timer.h"

#undef PARANOIA // for print VecAdd results

static void
usage(void)
{
    printf(
"mb_stream <options>\n"
"<options>:\n"
"  -b <blocks_per_grid>: number of TB per Grid\n"
"  -t <threads_per_block>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread(default: sizeof(int))\n"
"  -l <loop count>: tail GPU loop(default: 1)\n"
"  -p <# of partition>: number of memory partitions\n"
"  -u: allocate memory with uvm (cudaMallocManaged)\n"
"  -q: quiet\n"
"  -h: help\n");
}

static unsigned threads_per_block = 1;
static unsigned blocks_per_grid = 1;
static unsigned	io_size_per_thread = sizeof(int);
static unsigned n_loops = 1;
static unsigned	partitions = 0;
static int	need_uvm = 0;
static int	quiet;

// Device code
__global__ void
set_array_gpu(int *a, int value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) a[idx] = value;
}

__global__ void
STREAM_Copy(int *a, int *b, unsigned io_count_per_thread)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * io_count_per_thread;
    for (int i = 0; i < io_count_per_thread; i++)
        b[idx + i] = a[idx + i];
}

__global__ void
STREAM_Scale(int *a, int *b, int scale, unsigned io_count_per_thread)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * io_count_per_thread;
    for (int i = 0; i < io_count_per_thread; i++)
        b[idx + i] = scale * a[idx + i];
}

__global__ void
STREAM_Add(int *a, int *b, int *c, unsigned io_count_per_thread)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * io_count_per_thread;
    for (int i = 0; i < io_count_per_thread; i++)
        c[idx + i] = a[idx + i] + b[idx + i];
}

__global__ void
STREAM_Triad(int *a, int *b, int *c, int scalar, unsigned io_count_per_thread)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * io_count_per_thread;
    for (int i = 0; i < io_count_per_thread; i++)
        c[idx + i] = a[idx + i] + scalar * b[idx + i];
}

void
set_array_cpu(int *a, int value, unsigned len)
{
    for (unsigned idx = 0; idx < len; idx++) {
        a[idx] = value;
    }
}

// parse user input
static void
parse_args(int argc, char *argv[])
{
    int	c;

    while ((c = getopt(argc, argv, "b:t:s:l:p:uhq")) != -1) {
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
            case 'l':
                n_loops = mb_parse_count(optarg, "loop count");
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
main(int argc, char *argv[]) {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    unsigned ticks, i, j;
    unsigned long n_threads, io_count_per_thread;
    int   scalar = 3;
    size_t  total_io_size;
    cudaStream_t *streams;

    parse_args(argc, argv);
    if (!quiet) {
        char *str_io_size_per_thread = mb_get_sizestr(io_size_per_thread);

        printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %s, loops: %d, partitions: %d\n", threads_per_block,
               blocks_per_grid, str_io_size_per_thread, n_loops, partitions);
        free(str_io_size_per_thread);
    }

    n_threads = (unsigned long) threads_per_block * blocks_per_grid;
    total_io_size = (size_t) n_threads * io_size_per_thread;
    io_count_per_thread = (unsigned long) io_size_per_thread / sizeof(int);
    if (!quiet) {
        char	*str_memsize = mb_get_sizestr(total_io_size);
        printf("Managed memory used: %s\n", str_memsize);
        free(str_memsize);
    }

    if (need_uvm) {
        CUDA_CHECK(cudaMallocManaged((void **) &A, total_io_size), "cudaMallocManaged A");
        CUDA_CHECK(cudaMallocManaged((void **) &B, total_io_size), "cudaMallocManaged B");
        CUDA_CHECK(cudaMallocManaged((void **) &C, total_io_size), "cudaMallocManaged C");
    } else {
        CUDA_CHECK(cudaMalloc((void **) &d_A, total_io_size), "cudaMalloc A");
        CUDA_CHECK(cudaMalloc((void **) &d_B, total_io_size), "cudaMalloc B");
        CUDA_CHECK(cudaMalloc((void **) &d_C, total_io_size), "cudaMalloc C");

        if (partitions) {
            CUDA_CHECK(cudaMallocHost(&A, total_io_size), "cudaMallocHost A");
            CUDA_CHECK(cudaMallocHost(&B, total_io_size), "cudaMallocHost B");
            CUDA_CHECK(cudaMallocHost(&C, total_io_size), "cudaMallocHost C");
        } else {
            A = (int *) malloc(total_io_size);
            B = (int *) malloc(total_io_size);
            C = (int *) malloc(total_io_size);
        }
    }

    init_tickcount();

    set_array_cpu(A, 2, n_threads * io_count_per_thread);
    set_array_cpu(B, 5, n_threads * io_count_per_thread);
    set_array_cpu(C, 5, n_threads * io_count_per_thread);

    if (need_uvm) {
        for (j = 0; j < n_loops; j++)
        {
            STREAM_Copy<<<blocks_per_grid, threads_per_block>>>(A, C, io_count_per_thread);
            cudaDeviceSynchronize();

            STREAM_Scale<<<blocks_per_grid, threads_per_block>>>(B, C, scalar, io_count_per_thread);
            cudaDeviceSynchronize();

            STREAM_Add<<<blocks_per_grid, threads_per_block>>>(A, B, C, io_count_per_thread);
            cudaDeviceSynchronize();

            STREAM_Triad<<<blocks_per_grid, threads_per_block>>>(B, C, A, scalar, io_count_per_thread);
            cudaDeviceSynchronize();
        }
    }
    else if (partitions) {
        unsigned long	n_threads_part = n_threads / partitions;
        unsigned long	io_size_part = (unsigned long)total_io_size / partitions;
        unsigned long	offset = 0;
        streams = (cudaStream_t *)malloc(partitions * sizeof(cudaStream_t));

        for (i = 0; i < partitions; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]), "cudaStreamCreate");

            CUDA_CHECK(cudaMemcpyAsync(&d_A[offset], &A[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpyAsync A");
            CUDA_CHECK(cudaMemcpyAsync(&d_B[offset], &B[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpyAsync B");
            CUDA_CHECK(cudaMemcpyAsync(&d_C[offset], &C[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpyAsync C");

            for (j = 0; j < n_loops; j++)
            {
                STREAM_Copy<<<blocks_per_grid / partitions, threads_per_block, 0, streams[i]>>>(d_A + offset, d_C + offset, io_count_per_thread);
                cudaStreamSynchronize(streams[i]);

                STREAM_Scale<<<blocks_per_grid / partitions, threads_per_block, 0, streams[i]>>>(d_B + offset, d_C + offset, scalar, io_count_per_thread);
                cudaStreamSynchronize(streams[i]);

                STREAM_Add<<<blocks_per_grid / partitions, threads_per_block, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, io_count_per_thread);
                cudaStreamSynchronize(streams[i]);

                STREAM_Triad<<<blocks_per_grid / partitions, threads_per_block, 0, streams[i]>>>(d_B + offset, d_C + offset, d_A + offset, scalar, io_count_per_thread);
                cudaStreamSynchronize(streams[i]);
            }

            offset += n_threads_part * io_count_per_thread;
        }
    }
    else {
        CUDA_CHECK(cudaMemcpy(d_A, A, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy A");
        CUDA_CHECK(cudaMemcpy(d_B, B, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy B");
        CUDA_CHECK(cudaMemcpy(d_C, C, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy C");

        for (j = 0; j < n_loops; j++)
        {
            STREAM_Copy<<<blocks_per_grid, threads_per_block>>>(d_A, d_C, io_count_per_thread);
            cudaDeviceSynchronize();

            STREAM_Scale<<<blocks_per_grid, threads_per_block>>>(d_B, d_C, scalar, io_count_per_thread);
            cudaDeviceSynchronize();

            STREAM_Add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, io_count_per_thread);
            cudaDeviceSynchronize();

            STREAM_Triad<<<blocks_per_grid, threads_per_block>>>(d_B, d_C, d_A, scalar, io_count_per_thread);
            cudaDeviceSynchronize();
        }
    }

    ticks = get_tickcount();

// print result
#ifdef PARANOIA
for (i = 0; i < n_threads * io_count_per_thread; i++) {
	if (i % threads_per_block == 0)
		printf("\n");
	printf("%d:%d/ ", i, C[i]);
}
printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %d, loops: %d, partitions: %d\n", threads_per_block, blocks_per_grid, io_size_per_thread, n_loops, partitions);
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

        if (partitions) {
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(C);
        }
        else {
            free(A);
            free(B);
            free(C);
        }
    }

    if (partitions) {
        for (i = 0; i < partitions; i++)
            cudaStreamDestroy(streams[i]);
    }

    printf("elapsed: %.3f\n", ticks / 1000.0);
    return 0;
}