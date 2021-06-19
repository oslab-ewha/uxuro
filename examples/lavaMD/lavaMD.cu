#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <helper_cuda.h>

//#include "./util/num/num.h"

#include "lavaMD.h"
#include "cudaio.h"
#include "timer.h"

#include "kernel_gpu_cuda.cu"

static void
kernel_gpu_cuda_wrapper(par_str par_cpu, dim_str dim_cpu,
			cuio_ptr_t ptr_box, cuio_ptr_t ptr_rv,
			cuio_ptr_t ptr_qv, cuio_ptr_t ptr_fv)
{
	dim3	threads;
	dim3	blocks;

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;			// define the number of threads in the block
	threads.y = 1;

	// launch kernel - all boxes
	kernel_gpu_cuda<<<blocks, threads>>>(par_cpu, dim_cpu, (box_str *)ptr_box.ptr_d, (FOUR_VECTOR *)ptr_rv.ptr_d, (fp_t *)ptr_qv.ptr_d, (FOUR_VECTOR *)ptr_fv.ptr_d);
	getLastCudaError("kernel_gpu_cuda() execution failed\n");
	checkCudaErrors(cudaDeviceSynchronize());
}

static void
confer_load(FILE *fp, const char *fpath, void *ctx)
{
	char	buf[1024];
	int	*pn_boxes = (int *)ctx;

	if (fgets(buf, 1024, fp) == NULL) {
		fprintf(stderr, "cannot get # of boxes: %s\n", fpath);
		exit(2);
	}
	if (sscanf(buf, "%d", pn_boxes) != 1) {
		fprintf(stderr, "invalid format: %s\n", fpath);
		exit(3);
	}
}

int 
main(int argc, char *argv [])
{
	// system memory
	par_str	par_cpu;
	dim_str	dim_cpu;
	cuio_ptr_t	ptr_box;
	cuio_ptr_t	ptr_rv;
	cuio_ptr_t	ptr_qv;
	cuio_ptr_t	ptr_fv;
	unsigned	ticks_pre, ticks_kern, ticks_post;
	char	*folder;

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
		abort();
		exit(EXIT_FAILURE);
	}

	folder = argv[1];

	dim_cpu.cur_arg = 1;

	cuio_init(CUIO_TYPE_NONE, folder);
	cuio_load_conf(confer_load, &dim_cpu.boxes1d_arg);

	if (dim_cpu.boxes1d_arg < 0) {
		fprintf(stderr, "ERROR: wrong # of boxes configuration\n");
		abort();
		exit(EXIT_FAILURE);
	}

	// Print configuration
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	par_cpu.alpha = 0.5;

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp_t);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	init_tickcount();

	ptr_box = cuio_load_floats("box.mem", dim_cpu.box_mem / sizeof(float), CUIO_MODE_READONLY);
	ptr_rv = cuio_load_floats("rv.mem", dim_cpu.space_mem / sizeof(float), CUIO_MODE_READONLY);
	ptr_qv = cuio_load_floats("qv.mem", dim_cpu.space_elem, CUIO_MODE_READONLY);
	ptr_fv = cuio_load_floats("fv.mem", dim_cpu.space_mem / sizeof(float), CUIO_MODE_WRITEONLY);

	cuio_memcpy_h2d(&ptr_box);
	cuio_memcpy_h2d(&ptr_rv);
	cuio_memcpy_h2d(&ptr_qv);

	ticks_pre = get_tickcount();
	
	init_tickcount();
	kernel_gpu_cuda_wrapper(par_cpu, dim_cpu, ptr_box, ptr_rv, ptr_qv, ptr_fv);
	ticks_kern = get_tickcount();

	init_tickcount();

	cuio_memcpy_d2h(&ptr_fv);
	cuio_unload_floats("fv.mem", &ptr_fv);
	cuio_free_mem(&ptr_box);
	cuio_free_mem(&ptr_rv);
	cuio_free_mem(&ptr_qv);

	ticks_post = get_tickcount();

	printf("pre time(us): %u\n", ticks_pre);
	printf("kernel time(us): %u\n", ticks_kern);
	printf("post time(us): %u\n", ticks_post);

	return 0;
}
