#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "cudaio.h"
#include "timer.h"
#include "cuhelper.h"

#define BLOCK_SIZE	256
#define STR_SIZE	256
#define HALO		1 // halo width along one direction when advancing to the next iteration

#define IN_RANGE(x, min, max)		((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max)	x = (x < (min)) ? min : ((x > (max)) ? max: x )
#define MIN(a, b) ((a) <= (b) ? (a): (b))

__global__ void
dynproc_kernel(long iteration, int *gpuWall, int *gpuSrc,
	       int *gpuResults, long cols, long rows, long startStep, long border)
{
        __shared__ int	prev[BLOCK_SIZE];
        __shared__ int	result[BLOCK_SIZE];

	long	bx = (long)blockIdx.x;
	long	tx = (long)threadIdx.x;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	long	small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        long	blkX = small_block_cols * bx - border;
        long	blkXmax = blkX + BLOCK_SIZE - 1;

        // calculate the global thread coordination
	long	xidx = blkX + tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        long	validXmin = (blkX < 0) ? -blkX : 0;
        long	validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

        long	W = tx - 1;
        long	E = tx + 1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool	isValid = IN_RANGE(tx, validXmin, validXmax);

	if (IN_RANGE(xidx, 0, cols - 1)) {
		prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012

	bool computed;
        for (long i = 0; i < iteration; i++) { 
		computed = false;
		if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
			computed = true;
			long	left = prev[W];
			long	up = prev[tx];
			long	right = prev[E];
			long	shortest = MIN(left, up);
			shortest = MIN(shortest, right);
			long index = cols*(startStep+i) + xidx;
			result[tx] = shortest + gpuWall[index];
		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed)  //Assign the computation range
			prev[tx] = result[tx];
		__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the 
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		gpuResults[xidx] = result[tx];		
	}
}

/*
   compute N time steps
*/
static int
calc_path(cuio_ptr_t ptr_data, cuio_ptr_t ptr_res, unsigned size,
	  long pyramid_height, long blockCols, long borderCols)
{
	int	*gpuWall, *gpuResults[2];
        dim3	dimBlock(BLOCK_SIZE);
        dim3	dimGrid(blockCols);
        int src = 1, dst = 0;

	gpuWall = (int *)ptr_data.ptr_d + size;
	gpuResults[0] = (int *)ptr_data.ptr_d;
	gpuResults[1] = (int *)ptr_res.ptr_d;

	for (long t = 0; t < size - 1; t += pyramid_height) {
		int	temp = src;
		src = dst;
		dst = temp;

		dynproc_kernel<<<dimGrid, dimBlock>>>(MIN(pyramid_height, size - t - 1),
						      gpuWall, gpuResults[src], gpuResults[dst],
						      size, size, t, borderCols);
		CUDA_CALL_SAFE(cudaDeviceSynchronize());
	}
        return dst;
}

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	unsigned	*psize = (unsigned *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot get # of boxes: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%u", psize) != 1) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

int
main(int argc, char *argv[])
{
	cuio_ptr_t	ptr_data, ptr_res;
	const char	*folder;
	unsigned	size;
	unsigned	ticks_pre, ticks_kern, ticks_post;

	/* --------------- pyramid parameters --------------- */
	unsigned	pyramid_height = 1;
	unsigned	borderCols = pyramid_height * HALO;
	unsigned	smallBlockCol = BLOCK_SIZE - (pyramid_height) * HALO * 2;
	unsigned	blockCols;

	if (argc == 2) {
		folder = argv[1];
	}
	else {
		printf("Usage: %s <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &size);

	blockCols = size / smallBlockCol+((size % smallBlockCol == 0) ? 0: 1);
	printf("pyramidHeight: %u\ngridSize: [%u]\nborder:[%u]\nblockSize: %u\nblockGrid:[%u]\ntargetBlock:[%u]\n",
	       pyramid_height, size, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

	init_tickcount();

	ptr_data = cuio_load_ints("data.mem", size * size, CUIO_MODE_READONLY);
	ptr_res = cuio_load_ints("result.mem", size, CUIO_MODE_WRITEONLY);

	cuio_memcpy_h2d(&ptr_data);

	ticks_pre = get_tickcount();

	init_tickcount();
	calc_path(ptr_data, ptr_res, size, pyramid_height, blockCols, borderCols);
	ticks_kern = get_tickcount();

	init_tickcount();
	cuio_memcpy_d2h(&ptr_res);
	cuio_unload_ints("result.mem", &ptr_res);
	cuio_free_mem(&ptr_data);
	ticks_post = get_tickcount();
	
	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u\n", ticks_kern);
	printf("post time(us): %u\n", ticks_post);

	return 0;
}
