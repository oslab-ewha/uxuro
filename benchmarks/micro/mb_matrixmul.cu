#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>

#include "mb_common.h"
#include "timer.h"

# ifndef MIN
# define MIN(x, y) ((x) < (y) ? (x) : (y))
# endif

#undef PARANOIA // for print MatMul results

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
MatMul(int *A, int *B, int *C, unsigned M, unsigned K, unsigned N)
{
    /* A: MxK   B: KxN  C: MxN */
    unsigned i = (threadIdx.y + blockIdx.y * blockDim.y);
    unsigned j = (threadIdx.x + blockIdx.x * blockDim.x);

    if(i < M && j < N) {
        int elem = 0;
        for (unsigned k = 0; k < K; ++k)
            elem += A[i * K + k] * B[k * N + j];

        C[i * N + j] = elem;
    }
}

static int
get_random_number(int min, int max)
{
    int random = 0;

    while (1) {
        static unsigned int seed = 5323;
        seed = 8253729 * seed + 2396403;            // using overflow

        random = seed % max;

        if (random >= min && random < max)
            break;
    }
    return random;
}

static void
init_random_matrix(int *mat, unsigned row, unsigned col)
{
    for (unsigned i = 0; i < row; ++i) {
        for (unsigned j = 0; j < col; ++j) {
            int r = get_random_number(0, 10);
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
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    unsigned long   size;
    unsigned    ticks;

    parse_args(argc, argv);

    dim3    threads_per_block_dim(threads_per_block, threads_per_block);
    dim3    blocks_per_grid_dim(m / threads_per_block, n / threads_per_block);

    if (!quiet) {
        char	*str_threadsPerBlock = mb_get_sizestr(threads_per_block);
        char	*str_blocksPerGridX = mb_get_sizestr(blocks_per_grid_dim.x);
        char	*str_blocksPerGridY = mb_get_sizestr(blocks_per_grid_dim.y);
        char	*str_M = mb_get_sizestr(m);
        char	*str_K = mb_get_sizestr(k);
        char	*str_N = mb_get_sizestr(n);

        printf("threads_per_block_dim: (%s,%s), blocks_per_grid_dim: (%s,%s), m: %s, k: %s, n: %s\n",
               str_threadsPerBlock, str_threadsPerBlock, str_blocksPerGridX, str_blocksPerGridY, str_M, str_K, str_N);

        free(str_threadsPerBlock);
        free(str_blocksPerGridX);
        free(str_blocksPerGridY);
        free(str_M);
        free(str_K);
        free(str_N);
    }

    size = (unsigned long)m * k * n;
    if (!quiet) {
        char	*str_memsize = mb_get_sizestr(size);
        printf("Managed memory used: %s\n", str_memsize);
        free(str_memsize);
    }

    if (need_uvm) {
        CUDA_CHECK(cudaMallocManaged((void **) &A, m * k * sizeof(int)), "cudaMallocManaged A");
        CUDA_CHECK(cudaMallocManaged((void **) &B, k * n * sizeof(int)), "cudaMallocManaged B");
        CUDA_CHECK(cudaMallocManaged((void **) &C, m * n * sizeof(int)), "cudaMallocManaged C");
    } else {
        CUDA_CHECK(cudaMalloc((void **) &d_A, m * k * sizeof(int)), "cudaMalloc A");
        CUDA_CHECK(cudaMalloc((void **) &d_B, k * n * sizeof(int)), "cudaMalloc B");
        CUDA_CHECK(cudaMalloc((void **) &d_C, m * n * sizeof(int)), "cudaMalloc C");

        A = (int *) malloc(m * k * sizeof(int));
        B = (int *) malloc(k * n * sizeof(int));
        C = (int *) malloc(m * n * sizeof(int));
    }

    init_tickcount();

    init_random_matrix(A, m, k);
    init_random_matrix(B, k, n);

    if (need_uvm) {
        MatMul<<<blocks_per_grid_dim, threads_per_block_dim>>>(A, B, C, m, k, n);
        cudaDeviceSynchronize();
    } else {
        CUDA_CHECK(cudaMemcpy(d_A, A, m * k * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy A");
        CUDA_CHECK(cudaMemcpy(d_B, B, k * n * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy B");
        MatMul<<<blocks_per_grid_dim, threads_per_block_dim>>>(d_A, d_B, d_C, m, k, n);
        CUDA_CHECK(cudaMemcpy(C, d_C, m * n * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy C");
        cudaDeviceSynchronize();
    }

    ticks = get_tickcount();

// print result
#ifdef PARANOIA
    for (i = 0; i < m * k; i++) {
        if (i % threads_per_block == 0)
            printf("\n");
        printf("%d:%d/ ", i, A[i]);
    }
    printf("\n");

    for (i = 0; i < k * n; i++) {
        if (i % threads_per_block == 0)
            printf("\n");
        printf("%d:%d/ ", i, B[i]);
    }
    printf("\n");

    for (i = 0; i < m * n; i++) {
            if (i % threads_per_block == 0)
                printf("\n");
            printf("%d:%d/ ", i, C[i]);
    }
    printf("\nthreads_per_block_dim: (%d,%d), blocks_per_grid_dim: (%d,%d), m: %d, k: %d, n: %d\n",
               threads_per_block_dim.x, threads_per_block_dim.y, blocks_per_grid_dim.x, blocks_per_grid_dim.y, m, k, n);
#endif

    if (need_uvm) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    } else {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(A);
        free(B);
        free(C);
    }

    printf("elapsed: %.3f\n", ticks / 1000.0);
    return 0;
}
