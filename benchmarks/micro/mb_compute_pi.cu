#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>

#include "mb_common.h"
#include "timer.h"

#undef PARANOIA // for print results

static void
usage(void)
{
    printf(
"mb_compute_pi <options>\n"
"<options>:\n"
"  -b <blocks_per_grid>: number of TB per Grid\n"
"  -t <threads_per_block>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread(default: sizeof(double))\n"
"  -r <sample ratio>: ration of access count per array length\n"
"  -u: allocate memory with uvm (cudaMallocManaged)\n"
"  -q: quiet\n"
"  -h: help\n");
}

static unsigned threads_per_block = 1;
static unsigned blocks_per_grid = 1;
static unsigned	io_size_per_thread = sizeof(double);
static unsigned ratio = 10;
static int	need_uvm = 0;
static int	quiet;

__global__ void
circle_or_square(double* x, double* y, unsigned* indexes, int* iscircle, unsigned io_count_per_thread, unsigned access_count)
{
    double  distance;
    unsigned	idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned	jdx = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned    access_index = idx * io_count_per_thread + jdx;

    for (unsigned i = 0; i < access_count; i++) {
        if (access_index == indexes[i]) {
            distance = (x[access_index] * x[access_index]) + (y[access_index] * y[access_index]);

            if (distance <= 1)
                iscircle[i] = 1;
            else
                iscircle[i] = 0;
            break;
        }
    }
}

static double
cal_pi(int* iscircle, unsigned access_count)
{
    unsigned circle_point = 0;

    for (unsigned i = 0; i < access_count; i++){
        if (iscircle[i] == 1)
            circle_point++;
    }
    printf("circle_point: %d access_count: %d\n", circle_point, access_count);

    return (double)(4 * circle_point) / access_count;
}

static int
get_random_number(int min, int max)
{
    unsigned random;

    srand(clock());
    random = rand() % (max - min) + min;

    return random;
}

static void
set_random_index(unsigned *indexes, unsigned length, unsigned access_count)
{
    unsigned   *random;
    unsigned   i, j;
    unsigned   temp;
    unsigned   random_index;

    random = (unsigned *)malloc(length * sizeof(unsigned));
    // initialize
    for(i = 0; i < length; i++) {
        random[i] = 0;
    }

    // get random index
    for(i = 0; i < access_count; i++) {
        temp = get_random_number(0, length - 1);
        if (random[temp] != 1)
            random[temp] = 1;
        else
            i--;
    }

    // set index vector
    j = 0;
    for(i = 0; i < length; i++) {
        if (random[i] == 1) {
            indexes[j] = i;
            j++;
        }
        else
            continue;
    }

    // shuffle
    for(i = 0; i < access_count; i++) {
        temp = indexes[i];
        random_index = get_random_number(0, access_count);

        indexes[i] = indexes[random_index];
        indexes[random_index] = temp;
    }

    free(random);
}

// parse user input
static void
parse_args(int argc, char *argv[])
{
    int	c;

    while ((c = getopt(argc, argv, "b:t:s:r:p:uhq")) != -1) {
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
            case 'r':
                ratio = mb_parse_count(optarg, "access ratio");
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

    if (io_size_per_thread < sizeof(double)) {
        ERROR("IO size should be larger than sizeof(double)");
    }
}

int
main(int argc, char *argv[])
{
    int *iscircle, *d_iscircle;
    double  *x, *y;
    double  *d_x, *d_y;
    double  pi;
    unsigned    *indexes, *d_indexes;
    unsigned    ticks, i;
    unsigned    n_threads, io_count_per_thread;
    unsigned    access_count;
    size_t  total_io_size;
    size_t  total_io_count;

    parse_args(argc, argv);
    if (!quiet) {
        char	*str_io_size_per_thread = mb_get_sizestr(io_size_per_thread);

        printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %s, access ratio: %d\n", threads_per_block, blocks_per_grid, str_io_size_per_thread, ratio);
        free(str_io_size_per_thread);
    }

    n_threads = (unsigned)threads_per_block * blocks_per_grid;
    io_count_per_thread = (unsigned)io_size_per_thread / sizeof(double);
    total_io_size = (size_t)n_threads * io_size_per_thread;
    total_io_count = (size_t)n_threads * io_count_per_thread;
    access_count = (total_io_count) * ratio / 100;

    dim3    threads_per_block_dim(threads_per_block, 1);
    dim3    blocks_per_grid_dim(blocks_per_grid, io_count_per_thread);

    if (!quiet) {
        char	*str_memsize = mb_get_sizestr(total_io_size);
        printf("Managed memory used: %s\n", str_memsize);
        free(str_memsize);
    }

    if (need_uvm) {
        CUDA_CHECK(cudaMallocManaged((void **)&x, total_io_size), "cudaMallocManaged x");
        CUDA_CHECK(cudaMallocManaged((void **)&y, total_io_size), "cudaMallocManaged y");
        CUDA_CHECK(cudaMallocManaged((void **)&indexes, access_count * sizeof(unsigned)), "cudaMallocManaged indexes");
        CUDA_CHECK(cudaMallocManaged((void **)&iscircle, access_count * sizeof(int)), "cudaMallocManaged iscircle");
    }
    else {
        CUDA_CHECK(cudaMalloc((void **)&d_x, total_io_size), "cudaMalloc x");
        CUDA_CHECK(cudaMalloc((void **)&d_y, total_io_size), "cudaMalloc y");
        CUDA_CHECK(cudaMalloc((void **)&d_indexes, access_count * sizeof(unsigned)), "cudaMalloc indexes");
        CUDA_CHECK(cudaMalloc((void **)&d_iscircle, access_count * sizeof(int)), "cudaMalloc iscircle");

        x = (double *)malloc(total_io_size);
        y = (double *)malloc(total_io_size);
        indexes = (unsigned *)malloc(access_count * sizeof(unsigned));
        iscircle = (int *)malloc(access_count * sizeof(int));
    }

    set_random_index(indexes, n_threads * io_count_per_thread, access_count);

    init_tickcount();

    for (i = 0; i < total_io_count; i++) {
        x[i] = (double)get_random_number(0, total_io_count) / (total_io_count);
        y[i] = (double)get_random_number(0, total_io_count) / (total_io_count);
    }

    if (need_uvm) {
        circle_or_square<<<blocks_per_grid_dim, threads_per_block_dim>>>(x, y, indexes, iscircle, io_count_per_thread, access_count);
        cudaDeviceSynchronize();
    }
    else {
        CUDA_CHECK(cudaMemcpy(d_x, x, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy x");
        CUDA_CHECK(cudaMemcpy(d_y, y, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy y");
        CUDA_CHECK(cudaMemcpy(d_indexes, indexes, access_count * sizeof(unsigned), cudaMemcpyHostToDevice), "cudaMemcpy indexes");
        cudaDeviceSynchronize();

        circle_or_square<<<blocks_per_grid_dim, threads_per_block_dim>>>(d_x, d_y, d_indexes, d_iscircle, io_count_per_thread, access_count);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(iscircle, d_iscircle, access_count * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy iscircle");
        cudaDeviceSynchronize();
    }

    pi = cal_pi(iscircle, access_count);
    printf("Pi: %lf\n", pi);

    ticks = get_tickcount();

// print result
#ifdef PARANOIA
    for(i = 0; i < access_count; i++)
        printf("X[%d]=%lf, Y[%d]=%lf / ", indexes[i], x[indexes[i]], indexes[i], y[indexes[i]]);
#endif

    if (need_uvm) {
        cudaFree(x);
        cudaFree(y);
        cudaFree(indexes);
        cudaFree(iscircle);
    }
    else {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_indexes);
        cudaFree(d_iscircle);

        free(x);
        free(y);
        free(indexes);
        free(iscircle);
    }

    printf("elapsed: %.3f\n", ticks / 1000.0);
    return 0;
}