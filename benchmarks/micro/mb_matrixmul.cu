#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>

#include "mb_common.h"
#include "timer.h"

# ifndef MIN
# define MIN(x, y) ((x) < (y) ? (x) : (y))
# endif

#undef PARANOIA // for print matrix_mul results

static void
usage(void)
{
    printf(
"mb_matrixmul <options>\n"
"<options>:\n"
"  -m <row_of_multiplicand>: number of row in a [multiplicand/result] matrix\n"
"  -k <row_of_multiplier>: number of [col/row] in a [multiplicand/multiplier] matrix\n"
"  -n <col_of_result>: number of col in a [multiplier/result] matrix\n"
"  -t <threads_per_block>: number of threads per TB\n"
"  -u: allocate memory with uvm (cudaMallocManaged)\n"
"  -q: quiet\n"
"  -h: help\n");
}

static unsigned m = 1;
static unsigned k = 1;
static unsigned	n = 1;
static unsigned threads_per_block = 1;
static int	need_uvm = 0;
static int	quiet;

// Device code
__global__ void
matrix_mul(int *a, int *b, int *c, unsigned m, unsigned k, unsigned n)
{
    /* a: mxk   b: kxn  c: mxn */
    unsigned idx = (threadIdx.y + blockIdx.y * blockDim.y);
    unsigned jdx = (threadIdx.x + blockIdx.x * blockDim.x);

    if(idx < m && jdx < n) {
        int elem = 0;
        for (unsigned kdx = 0; kdx < k; ++kdx)
            elem += a[idx * k + kdx] * b[kdx * n + jdx];

        c[idx * n + jdx] = elem;
    }
}

static void
read_value_from_cpu(int* mem, unsigned length) {
    int value;

    for (unsigned i = 0; i < length; i++) {
        value = mem[i];
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
init_random_matrix(int *mat, unsigned row, unsigned col)
{
    int r;
    for (unsigned i = 0; i < row; ++i) {
        for (unsigned j = 0; j < col; ++j) {
            r = get_random_number(0, 10);
            mat[i * col + j] = r;
        }
    }
}

// parse user input
static void
parse_args(int argc, char *argv[])
{
    int	c;

    while ((c = getopt(argc, argv, "m:k:n:t:uhq")) != -1) {
        switch (c) {
            case 'm':
                m = mb_parse_size(optarg, "row_of_multiplicand");
                break;
            case 'k':
                k = mb_parse_size(optarg, "row_of_multiplier");
                break;
            case 'n':
                n = mb_parse_size(optarg, "col_of_result");
                break;
            case 't':
                threads_per_block = mb_parse_count(optarg, "threads_per_block");
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

    if (MIN(n, MIN(m, k)) < threads_per_block) {
        ERROR("threads_per_block must be lower than (row, col) of matrices");
    }
}

int
main(int argc, char *argv[])
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    unsigned long   size;
    unsigned    ticks, i;

    parse_args(argc, argv);

    dim3    threads_per_block_dim(threads_per_block, threads_per_block);
    dim3    blocks_per_grid_dim(m / threads_per_block, n / threads_per_block);

    if (!quiet) {
        char	*str_threadsPerBlock = mb_get_sizestr(threads_per_block);
        char	*str_blocksPerGridX = mb_get_sizestr(blocks_per_grid_dim.x);
        char	*str_blocksPerGridY = mb_get_sizestr(blocks_per_grid_dim.y);
        char	*str_m = mb_get_sizestr(m);
        char	*str_k = mb_get_sizestr(k);
        char	*str_n = mb_get_sizestr(n);

        printf("threads_per_block_dim: (%s,%s), blocks_per_grid_dim: (%s,%s), m: %s, k: %s, n: %s\n",
               str_threadsPerBlock, str_threadsPerBlock, str_blocksPerGridX, str_blocksPerGridY, str_m, str_k, str_n);

        free(str_threadsPerBlock);
        free(str_blocksPerGridX);
        free(str_blocksPerGridY);
        free(str_m);
        free(str_k);
        free(str_n);
    }

    size = (unsigned long)m * k * n;
    if (!quiet) {
        char	*str_memsize = mb_get_sizestr(size);
        printf("Managed memory used: %s\n", str_memsize);
        free(str_memsize);
    }

    if (need_uvm) {
        CUDA_CHECK(cudaMallocManaged((void **) &a, m * k * sizeof(int)), "cudaMallocManaged a");
        CUDA_CHECK(cudaMallocManaged((void **) &b, k * n * sizeof(int)), "cudaMallocManaged b");
        CUDA_CHECK(cudaMallocManaged((void **) &c, m * n * sizeof(int)), "cudaMallocManaged c");
    } else {
        CUDA_CHECK(cudaMalloc((void **) &d_a, m * k * sizeof(int)), "cudaMalloc a");
        CUDA_CHECK(cudaMalloc((void **) &d_b, k * n * sizeof(int)), "cudaMalloc b");
        CUDA_CHECK(cudaMalloc((void **) &d_c, m * n * sizeof(int)), "cudaMalloc c");

        a = (int *) malloc(m * k * sizeof(int));
        b = (int *) malloc(k * n * sizeof(int));
        c = (int *) malloc(m * n * sizeof(int));
    }

    init_tickcount();

    init_random_matrix(a, m, k);
    init_random_matrix(b, k, n);

    if (need_uvm) {
        matrix_mul<<<blocks_per_grid_dim, threads_per_block_dim>>>(a, b, c, m, k, n);
        cudaDeviceSynchronize();
    } else {
        CUDA_CHECK(cudaMemcpy(d_a, a, m * k * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy a");
        CUDA_CHECK(cudaMemcpy(d_b, b, k * n * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy b");
        matrix_mul<<<blocks_per_grid_dim, threads_per_block_dim>>>(d_a, d_b, d_c, m, k, n);
        CUDA_CHECK(cudaMemcpy(c, d_c, m * n * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy c");
        cudaDeviceSynchronize();
    }

    read_value_from_cpu(c, m * n);

    ticks = get_tickcount();

// print result
#ifdef PARANOIA
    for (i = 0; i < m * k; i++) {
        if (i % k == 0)
            printf("\n");
        printf("%d:%d/ ", i, a[i]);
    }
    printf("\n");

    for (i = 0; i < k * n; i++) {
        if (i % n == 0)
            printf("\n");
        printf("%d:%d/ ", i, b[i]);
    }
    printf("\n");

    for (i = 0; i < m * n; i++) {
        if (i % n == 0)
            printf("\n");
        printf("%d:%d/ ", i, c[i]);
    }
    printf("\nthreads_per_block_dim: (%d,%d), blocks_per_grid_dim: (%d,%d), m: %d, k: %d, n: %d\n",
               threads_per_block_dim.x, threads_per_block_dim.y, blocks_per_grid_dim.x, blocks_per_grid_dim.y, m, k, n);
#endif

    if (need_uvm) {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    } else {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        free(a);
        free(b);
        free(c);
    }

    printf("elapsed: %.3f\n", ticks / 1000.0);
    return 0;
}
