#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>

#include "mb_common.h"
#include "timer.h"

#undef PARANOIA

static void
usage(void)
{
    printf(
"mb_random <options>\n"
"<options>:\n"
"  -b <blocks_per_grid>: number of TB per Grid\n"
"  -t <threads_per_block>: number of threads per TB\n"
"  -s <IO size>: read/write size in byte per thread(default: sizeof(int))\n"
"  -r <access ratio>: ration of access count per array length\n"
"  -u: allocate memory with uvm (cudaMallocManaged)\n"
"  -q: quiet\n"
"  -h: help\n");
}

static unsigned threads_per_block = 1;
static unsigned blocks_per_grid = 1;
static unsigned	io_size_per_thread = sizeof(int);
static unsigned ratio = 10;
static int	need_uvm = 0;
static int	quiet;

// Device code
__global__ void
random_scalar_mul(int* a, int* b, unsigned* indexes, int scale, unsigned access_count)
{
    unsigned	idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (unsigned i = 0; i < access_count; i++) {
        if (idx == indexes[i]) {
            b[i] = scale * a[idx];
            break;
        }
    }
}

static int
get_random_number(int min, int max)
{
    int random;

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

static unsigned long long
reduce_sum(int* mem, unsigned length) {
    unsigned long long   sum = 0;

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

    if (io_size_per_thread < sizeof(int)) {
        ERROR("IO size should be larger than sizeof(int)");
    }
}

int
main(int argc, char *argv[])
{
    int	*a, *b;
    int	*d_a, *d_b;
    unsigned	ticks, i;
    unsigned    scale;
    unsigned    *indexes, *d_indexes;
    unsigned    access_count;
    unsigned    n_threads, io_count_per_thread;
    unsigned long long  sum;
    size_t  total_io_size;

    parse_args(argc, argv);
    if (!quiet) {
        char	*str_io_size_per_thread = mb_get_sizestr(io_size_per_thread);

        printf("threads_per_block: %d, blocks_per_grid: %d, IO_size: %s, access ratio: %d\n", threads_per_block, blocks_per_grid, str_io_size_per_thread, ratio);
        free(str_io_size_per_thread);
    }

    n_threads = (unsigned)threads_per_block * blocks_per_grid;
    total_io_size = (size_t)n_threads * io_size_per_thread;
    io_count_per_thread = (unsigned)io_size_per_thread / sizeof(int);
    access_count = (n_threads * io_count_per_thread) * ratio / 100;
    scale = 3;

    if (!quiet) {
        char	*str_memsize = mb_get_sizestr(total_io_size);
        printf("Managed memory used: %s\n", str_memsize);
        free(str_memsize);
    }

    if (need_uvm) {
        CUDA_CHECK(cudaMallocManaged((void **)&a, total_io_size), "cudaMallocManaged a");
        CUDA_CHECK(cudaMallocManaged((void **)&b, access_count * sizeof(int)), "cudaMallocManaged b");
        CUDA_CHECK(cudaMallocManaged((void **)&indexes, access_count * sizeof(unsigned)), "cudaMallocManaged indexes");
    }
    else {
        CUDA_CHECK(cudaMalloc((void **)&d_a, total_io_size), "cudaMalloc a");
        CUDA_CHECK(cudaMalloc((void **)&d_b, access_count * sizeof(int)), "cudaMalloc b");
        CUDA_CHECK(cudaMalloc((void **)&d_indexes, access_count * sizeof(unsigned)), "cudaMalloc indexes");

        a = (int *)malloc(total_io_size);
        b = (int *)malloc(access_count * sizeof(int));
        indexes = (unsigned *)malloc(access_count * sizeof(unsigned));
    }

    set_random_index(indexes, n_threads * io_count_per_thread, access_count);

    init_tickcount();

    for (i = 0; i < n_threads * io_count_per_thread; i++) {
        a[i] = i % 1024;    // to avoid exceeding the integer range, limit the element value of 'vector a' to between 0 and 1023.
    }

    if (need_uvm) {
        random_scalar_mul<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(a, b, indexes, scale, access_count);
        cudaDeviceSynchronize();
    }
    else {
        CUDA_CHECK(cudaMemcpy(d_a, a, total_io_size, cudaMemcpyHostToDevice), "cudaMemcpy a");
        CUDA_CHECK(cudaMemcpy(d_indexes, indexes, access_count * sizeof(unsigned), cudaMemcpyHostToDevice), "cudaMemcpy indexes");
        cudaDeviceSynchronize();

        random_scalar_mul<<<blocks_per_grid * io_count_per_thread, threads_per_block>>>(d_a, d_b, d_indexes, scale, access_count);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(b, d_b, access_count * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy b");
        cudaDeviceSynchronize();
    }

    sum = reduce_sum(b, access_count);

    ticks = get_tickcount();

#ifdef PARANOIA
    printf("result_sum : %lld\n", sum);

    sum = 0;
    for (i = 0; i < access_count; i++) {
        sum += indexes[i] % 1024;
    }
    printf("ground_truth : %lld\n", sum * scale);
#endif

    if (need_uvm) {
        cudaFree(a);
        cudaFree(b);
        cudaFree(indexes);
    }
    else {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_indexes);

        free(a);
        free(b);
        free(indexes);
    }

    printf("elapsed: %.3f\n", ticks / 1000.0);
    return 0;
}