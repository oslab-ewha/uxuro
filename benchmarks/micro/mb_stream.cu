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
#include <sys/mman.h>

#define CUDA_API_PER_THREAD_DEFAULT_STEAM // to Overlap Data Transfers in CUDA Stream
#include "mb_common.h"
#include "timer.h"

#undef PARANOIA // for print results

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
set_array_gpu(int *mem, int value, int len)
{
    unsigned long   idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        mem[idx] = value;
}

__global__ void
STREAM_copy(int *a, int *b)
{
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    b[idx] = a[idx];
}

__global__ void
STREAM_scale(int *a, int *b, int scale)
{
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    b[idx] = scale * a[idx];
}

__global__ void
STREAM_add(int *a, int *b, int *c)
{
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    c[idx] = a[idx] + b[idx];
}

__global__ void
STREAM_triad(int *a, int *b, int *c, int scalar)
{
    unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    c[idx] = a[idx] + scalar * b[idx];
}

static void
set_array_cpu(int *mem, int value, unsigned length)
{
    for (unsigned i = 0; i < length; i++) {
        mem[i] = value;
    }
}

static unsigned
reduce_sum(int* mem, unsigned length) {
    unsigned long   sum = 0;

    for (unsigned i = 0; i < length; i++) {
        sum += mem[i];
    }

    return sum;
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
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int scale = 3, scalar = 5;
    unsigned ticks, loops, i;
    unsigned n_threads, io_count_per_thread;
    unsigned long long  sum_a, sum_b, sum_c;
    size_t  total_io_size;
    cudaStream_t    *streams;

    parse_args(argc, argv);
    if (!quiet) {
        char *str_io_size_per_thread = mb_get_sizestr(io_size_per_thread);

        printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %s, loops: %d, partitions: %d\n", threads_per_block,
               blocks_per_grid, str_io_size_per_thread, n_loops, partitions);
        free(str_io_size_per_thread);
    }

    n_threads = (unsigned)threads_per_block * blocks_per_grid;
    total_io_size = (size_t) n_threads * io_size_per_thread;
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
            a = (int *) malloc(total_io_size);
            b = (int *) malloc(total_io_size);
            c = (int *) malloc(total_io_size);
        }
    }

    init_tickcount();

    if (need_uvm) {
        for (loops = 0; loops < n_loops; loops++) {
            set_array_cpu(a, 2, n_threads * io_count_per_thread);
            set_array_cpu(b, 3, n_threads * io_count_per_thread);
            set_array_cpu(c, 5, n_threads * io_count_per_thread);
            
            STREAM_copy<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(a, c);
            cudaDeviceSynchronize();

            STREAM_scale<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(b, c, scale);
            cudaDeviceSynchronize();

            STREAM_add<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(a, b, c);
            cudaDeviceSynchronize();

            STREAM_triad<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(b, c, a, scalar);
            cudaDeviceSynchronize();
        }
    }
    else if (partitions) {
        unsigned	n_threads_part = n_threads / partitions;
        unsigned	io_size_part = (unsigned)total_io_size / partitions;
        unsigned    offset;
        streams = (cudaStream_t *)malloc(partitions * sizeof(cudaStream_t));

        for (loops = 0; loops < n_loops; loops++) {
            offset = 0;
            set_array_cpu(a, 2, n_threads * io_count_per_thread);
            set_array_cpu(b, 3, n_threads * io_count_per_thread);
            set_array_cpu(c, 5, n_threads * io_count_per_thread);

            for (i = 0; i < partitions; i++) {
                CUDA_CHECK(cudaStreamCreate(&streams[i]), "cudaStreamCreate");

                CUDA_CHECK(cudaMemcpyAsync(&d_a[offset], &a[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]),
                           "cudaMemcpyAsync a to device");
                CUDA_CHECK(cudaMemcpyAsync(&d_b[offset], &b[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]),
                           "cudaMemcpyAsync b to device");
                CUDA_CHECK(cudaMemcpyAsync(&d_c[offset], &c[offset], io_size_part, cudaMemcpyHostToDevice, streams[i]),
                           "cudaMemcpyAsync c to device");

                STREAM_copy<<<(blocks_per_grid * io_count_per_thread) / partitions, threads_per_block, 0, streams[i]>>>(
                        d_a + offset, d_c + offset);
                cudaStreamSynchronize(streams[i]);

                STREAM_scale<<<(blocks_per_grid * io_count_per_thread) / partitions, threads_per_block, 0, streams[i]>>>(
                        d_b + offset, d_c + offset, scale);
                cudaStreamSynchronize(streams[i]);

                STREAM_add<<<(blocks_per_grid * io_count_per_thread) / partitions, threads_per_block, 0, streams[i]>>>(
                        d_a + offset, d_b + offset, d_c + offset);
                cudaStreamSynchronize(streams[i]);

                STREAM_triad<<<(blocks_per_grid * io_count_per_thread) / partitions, threads_per_block, 0, streams[i]>>>(
                        d_b + offset, d_c + offset, d_a + offset, scalar);
                cudaStreamSynchronize(streams[i]);

                CUDA_CHECK(cudaMemcpyAsync(&a[offset], &d_a[offset], io_size_part, cudaMemcpyDeviceToHost, streams[i]),
                           "cudaMemcpyAsync a to host");
                CUDA_CHECK(cudaMemcpyAsync(&b[offset], &d_b[offset], io_size_part, cudaMemcpyDeviceToHost, streams[i]),
                           "cudaMemcpyAsync b to host");
                CUDA_CHECK(cudaMemcpyAsync(&c[offset], &d_c[offset], io_size_part, cudaMemcpyDeviceToHost, streams[i]),
                           "cudaMemcpyAsync c to host");
                cudaStreamSynchronize(streams[i]);

                offset += n_threads_part * io_count_per_thread;
            }
        }
    }
    else {
        for (loops = 0; loops < n_loops; loops++) {
            set_array_cpu(a, 2, n_threads * io_count_per_thread);
            set_array_cpu(b, 3, n_threads * io_count_per_thread);
            set_array_cpu(c, 5, n_threads * io_count_per_thread);

            CUDA_CHECK(cudaMemcpy(d_a, a, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy a to device");
            CUDA_CHECK(cudaMemcpy(d_b, b, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy b to device");
            CUDA_CHECK(cudaMemcpy(d_c, c, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy c to device");

            STREAM_copy<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(d_a, d_c);
            cudaDeviceSynchronize();

            STREAM_scale<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(d_b, d_c, scale);
            cudaDeviceSynchronize();

            STREAM_add<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(d_a, d_b, d_c);
            cudaDeviceSynchronize();

            STREAM_triad<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(d_b, d_c, d_a, scalar);
            cudaDeviceSynchronize();

            CUDA_CHECK(cudaMemcpy(a, d_a, total_io_size, cudaMemcpyDeviceToHost), "cudaMemcpy a to host");
            CUDA_CHECK(cudaMemcpy(b, d_b, total_io_size, cudaMemcpyDeviceToHost), "cudaMemcpy b to host");
            CUDA_CHECK(cudaMemcpy(c, d_c, total_io_size, cudaMemcpyDeviceToHost), "cudaMemcpy c to host");
        }
    }

    sum_a = reduce_sum(a, n_threads * io_count_per_thread);
    sum_b = reduce_sum(b, n_threads * io_count_per_thread);
    sum_c = reduce_sum(c, n_threads * io_count_per_thread);

    ticks = get_tickcount();

// print result
#ifdef PARANOIA
    printf("result_sum: %lld\n", sum_a + sum_b + sum_c);
    printf("ground_truth: %lld\n", ((3 + scalar * 5) + 3 + (2 + 3)) * (n_threads * io_count_per_thread));
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