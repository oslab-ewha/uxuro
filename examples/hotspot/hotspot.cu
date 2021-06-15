#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include "cudaio.h"
#include "timer.h"
#include "cuhelper.h"

#define BLOCK_SIZE 32

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
static float	t_chip = 0.0005;
static float	chip_height = 0.016;
static float	chip_width = 0.016;

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

static __global__ void
calculate_temp(long iteration,	//number of iteration
	       float *power,	//power input
	       float *temp_src,	//temperature input/output
	       float *temp_dst,	//temperature input/output
	       long grid_cols,	//Col of grid
	       long grid_rows,	//Row of grid
	       long border_cols,	// border offset
	       long border_rows,	// border offset
	       float Cap,		//Capacitance
	       float Rx, float Ry, float Rz,float step, float time_elapsed)
{
	__shared__ float	temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float	power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float	temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

	float	amb_temp = 80.0;
	float	step_div_Cap;
	float	Rx_1,Ry_1,Rz_1;

	long	bx = blockIdx.x;
	long	by = blockIdx.y;

	long	tx = threadIdx.x;
	long	ty = threadIdx.y;

	step_div_Cap = step / Cap;

	Rx_1 = 1 / Rx;
	Ry_1 = 1 / Ry;
	Rz_1 = 1 / Rz;

	// each block finally computes result for a small block
	// after N iterations. 
	// it is the non-overlapping small blocks that cover 
	// all the input data

	// calculate the small block size
	long	small_block_rows = BLOCK_SIZE - iteration * 2; //EXPAND_RATE
	long	small_block_cols = BLOCK_SIZE - iteration * 2; //EXPAND_RATE

	// calculate the boundary for the block according to 
	// the boundary of its small block
	long	blkY = small_block_rows * by - border_rows;
	long	blkX = small_block_cols * bx - border_cols;
	long	blkYmax = blkY + BLOCK_SIZE - 1;
	long	blkXmax = blkX + BLOCK_SIZE - 1;

	// calculate the global thread coordination
	long	yidx = blkY + ty;
	long	xidx = blkX + tx;

	// load data if it is within the valid input range
	long	loadYidx = yidx, loadXidx = xidx;
	long	index = grid_cols * loadYidx + loadXidx;

	if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
		temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
		power_on_cuda[ty][tx] = power[index];    // Load the power data from global memory to shared memory
	}
	__syncthreads();

	// effective range within this block that falls within 
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	long	validYmin = (blkY < 0) ? -blkY : 0;
	long	validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax-grid_rows + 1) : BLOCK_SIZE - 1;
	long	validXmin = (blkX < 0) ? -blkX : 0;
	long	validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax-grid_cols + 1) : BLOCK_SIZE - 1;

	long	N = ty - 1;
	long	S = ty + 1;
	long	W = tx - 1;
	long	E = tx + 1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	for (long i = 0; i < iteration; i++) {
		computed = false;
		if (IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&
		    IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&
		    IN_RANGE(tx, validXmin, validXmax) &&
		    IN_RANGE(ty, validYmin, validYmax)) {
			computed = true;
			temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
										(temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0 * temp_on_cuda[ty][tx]) * Ry_1 +
										(temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0 * temp_on_cuda[ty][tx]) * Rx_1 +
										(amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed)	 //Assign the computation range
			temp_on_cuda[ty][tx] = temp_t[ty][tx];
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the 
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		temp_dst[index]= temp_t[ty][tx];
	}
}

/*
 * compute N time steps
 */
static int
compute_tran_temp(cuio_ptr_t power, cuio_ptr_t temps[2], long col, long row,
		  long total_iterations, long num_iterations, long blockCols, long blockRows, long borderCols, long borderRows) 
{
        dim3	dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3	dimGrid(blockCols, blockRows);

	float	grid_height = chip_height / row;
	float	grid_width = chip_width / col;

	float	Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float	Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float	Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float	Rz = t_chip / (K_SI * grid_height * grid_width);

	float	max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float	step = PRECISION / max_slope;
	float	t;

	float	time_elapsed = 0.001;

	int src = 1, dst = 0;

	for (t = 0; t < total_iterations; t += num_iterations) {
		int	temp = src;

		src = dst;
		dst = temp;
		calculate_temp<<< dimGrid, dimBlock >>>(MIN(num_iterations, total_iterations - t),
							(float *)power.ptr_d,
							(float *)temps[src].ptr_d,
							(float *)temps[dst].ptr_d,
							col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
	}
	CUDA_CALL_SAFE(cudaDeviceSynchronize());
	return dst;
}

typedef struct {
	long	rows, cols;
} params_t;

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	params_t	*pparams = (params_t *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot option count: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%ld %ld", &pparams->rows, &pparams->cols) != 2) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

int
main(int argc, char *argv[])
{
	cuio_ptr_t	temps[2], power;
	size_t	size;
	params_t	params;
	const char	*folder;

	long	total_iterations = 60;
	long	pyramid_height = 1; // number of iterations

	unsigned	ticks_pre, ticks_kern, ticks_post;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	folder = argv[1];

	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &params);

	size = params.rows * params.cols;

	/* --------------- pyramid parameters --------------- */
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
	long	borderCols = (pyramid_height) * EXPAND_RATE / 2;
	long	borderRows = (pyramid_height) * EXPAND_RATE / 2;
	long	smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
	long	smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
	long	blockCols = params.cols / smallBlockCol + ((params.cols % smallBlockCol == 0) ? 0: 1);
	long	blockRows = params.rows / smallBlockRow + ((params.rows % smallBlockRow == 0) ? 0: 1);

	printf("pyramidHeight: %ld\ngridSize: [%ld, %ld]\nborder:[%ld, %ld]\nblockGrid:[%ld, %ld]\ntargetBlock:[%ld, %ld]\n", \
	       pyramid_height, params.cols, params.rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

	init_tickcount();

	temps[0] = cuio_load_floats("temperature", size, CUIO_MODE_READWRITE);
	temps[1] = cuio_load_floats("output", size, CUIO_MODE_WRITEONLY);
	power = cuio_load_floats("power", size, CUIO_MODE_READONLY);

	ticks_pre = get_tickcount();

	cuio_memcpy_h2d(&temps[0]);
	cuio_memcpy_h2d(&power);

	printf("Start computing the transient temperature\n");

	init_tickcount();
	int ret = compute_tran_temp(power, temps, params.cols, params.rows,
				    total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows);
	ticks_kern = get_tickcount();

	printf("Ending simulation\n");

	init_tickcount();
	cuio_memcpy_d2h(&temps[1]);

	cuio_unload_floats("output", &temps[1]);
	cuio_free_mem(&temps[0]);
	cuio_free_mem(&power);
	ticks_post = get_tickcount();

	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u\n", ticks_kern);
	printf("post time(us): %u\n", ticks_post);

	return 0;
}
