// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

#include "cudaio.h"
#include "timer.h"
#include "cuhelper.h"

#define R1	0	//y1 position of the speckle
#define R2	127	//y2 position of the speckle
#define C1	0	//x1 position of the speckle
#define C2	127	//x2 position of the speckle
#define LAMBDA	0.5	//Lambda value
#define NITER	2	//number of iterations

static unsigned	size, size_I, size_R;
static unsigned ticks_pre, ticks_cpu, ticks_gpu, ticks_post;

static void
calc_matrix(cuio_ptr_t ptr_J, cuio_ptr_t ptr_C, cuio_ptr_t ptr_C_E, cuio_ptr_t ptr_C_W, cuio_ptr_t ptr_C_N, cuio_ptr_t ptr_C_S)
{
	float	sum, sum2;
	float	meanROI, varROI, q0sqr;

	init_tickcount();
	sum = 0; sum2 = 0;

	for (long i = R1; i <= R2; i++) {
		for (long j = C1; j <= C2; j++) {
			float	tmp = ((float *)ptr_J.ptr_h)[i * size + j];

			sum  += tmp;
			sum2 += tmp * tmp;
		}
	}

	ticks_cpu += get_tickcount();

	init_tickcount();

	meanROI = sum / size_R;
	varROI  = (sum2 / size_R) - meanROI * meanROI;
	q0sqr   = varROI / (meanROI * meanROI);

	//Currently the input size must be divided by 16 - the block size
	long	block_x = size / (long)BLOCK_SIZE;
	long	block_y = size / (long)BLOCK_SIZE;

	dim3	dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3	dimGrid(block_x , block_y);

	//Copy data from main memory to device memory
	cuio_memcpy_h2d(&ptr_J);

	ticks_pre += get_tickcount();

	init_tickcount();
	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock>>>(CUIO_FLOATS_D(ptr_C_E), CUIO_FLOATS_D(ptr_C_W), CUIO_FLOATS_D(ptr_C_N), CUIO_FLOATS_D(ptr_C_S),
					   CUIO_FLOATS_D(ptr_J), CUIO_FLOATS_D(ptr_C), size, size, q0sqr);
	srad_cuda_2<<<dimGrid, dimBlock>>>(CUIO_FLOATS_D(ptr_C_E), CUIO_FLOATS_D(ptr_C_W), CUIO_FLOATS_D(ptr_C_N), CUIO_FLOATS_D(ptr_C_S),
					   CUIO_FLOATS_D(ptr_J), CUIO_FLOATS_D(ptr_C), size, size, LAMBDA, q0sqr);

	CUDA_CALL_SAFE(cudaDeviceSynchronize());
	ticks_gpu += get_tickcount();

	init_tickcount();
	//Copy data from device memory to main memory
	cuio_memcpy_d2h(&ptr_J);
	ticks_post += get_tickcount();
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
	cuio_ptr_t	ptr_J, ptr_C, ptr_C_E, ptr_C_W, ptr_C_N, ptr_C_S;
	char *folder;

	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
 
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
		exit(1);
	}
	 
	folder = argv[1];

	init_tickcount();

	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &size);

	size_I = size * size;
	size_R = (R2 - R1 + 1) * (C2 - C1 + 1);

	ptr_J = cuio_load_floats("matrix.mem", size_I, CUIO_MODE_READWRITE);
	ptr_C = cuio_load_floats("matrix.C", size_I, CUIO_MODE_WRITEONLY);
	ptr_C_E = cuio_load_floats("matrix.C.E", size_I, CUIO_MODE_WRITEONLY);
	ptr_C_W = cuio_load_floats("matrix.C.W", size_I, CUIO_MODE_WRITEONLY);
	ptr_C_N = cuio_load_floats("matrix.C.N", size_I, CUIO_MODE_WRITEONLY);
	ptr_C_S = cuio_load_floats("matrix.C.S", size_I, CUIO_MODE_WRITEONLY);

	ticks_pre += get_tickcount();

	printf("Start the SRAD main loop\n");
	for (int iter = 0; iter < NITER; iter++) {
		calc_matrix(ptr_J, ptr_C, ptr_C_E, ptr_C_W, ptr_C_N, ptr_C_S);
	}

	init_tickcount();

	cuio_unload_floats("matrix.mem", &ptr_J);
	cuio_free_mem(&ptr_C);
	cuio_free_mem(&ptr_C_E);
	cuio_free_mem(&ptr_C_W);
	cuio_free_mem(&ptr_C_N);
	cuio_free_mem(&ptr_C_S);

	ticks_post += get_tickcount();

	printf("Computation Done\n");

	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u(gpu:%u)\n", ticks_cpu + ticks_gpu, ticks_gpu);
	printf("post time(us): %u\n", ticks_post);

	return 0;
}
