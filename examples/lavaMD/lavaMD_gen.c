#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>

#include "lavaMD.h"
#include "cudaio.h"

static const char	*folder;

static void
set_box(box_str *box_cpu, int nh, int i, int j, int k, int boxsize)
{
	int	l, m, n;

	// current home box
	box_cpu[nh].x = k;
	box_cpu[nh].y = j;
	box_cpu[nh].z = i;
	box_cpu[nh].number = nh;
	box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

	// initialize number of neighbor boxes
	box_cpu[nh].nn = 0;

	// neighbor boxes in z direction
	for (l = -1; l < 2; l++) {
		// neighbor boxes in y direction
		for(m = -1; m < 2; m++) {
			// neighbor boxes in x direction
			for(n = -1; n < 2; n++) {
				// check if (this neighbor exists) and (it is not the same as home box)
				if (((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0 &&
				     (i + l) < boxsize && (j + m) < boxsize && (k + n) < boxsize) &&
				    !(l == 0 && m == 0 && n == 0)) {
					// current neighbor box
					box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
					box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
					box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
					box_cpu[nh].nei[box_cpu[nh].nn].number = (box_cpu[nh].nei[box_cpu[nh].nn].z * boxsize * boxsize) + 
						(box_cpu[nh].nei[box_cpu[nh].nn].y * boxsize) + 
						box_cpu[nh].nei[box_cpu[nh].nn].x;
					box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

					// increment neighbor box
					box_cpu[nh].nn = box_cpu[nh].nn + 1;
				}

			} // neighbor boxes in x direction
		} // neighbor boxes in y direction
	} // neighbor boxes in z direction
}

static void
fill_box_data(box_str *box_cpu, int boxsize)
{
	int	i, j, k;
	int	nh;

	nh = 0;

	// home boxes in z direction
	for (i = 0; i < boxsize; i++) {
		// home boxes in y direction
		for(j = 0; j < boxsize; j++) {
			// home boxes in x direction
			for (k = 0; k < boxsize; k++, nh++) {
				set_box(box_cpu, nh, i, j, k, boxsize);
			} // home boxes in x direction
		} // home boxes in y direction
	} // home boxes in z direction
}

static void
make_box_file(int n_boxes, int boxsize)
{
	box_str	*box_cpu;
	char	fpath[512];
	FILE	*fp;
	size_t	size = n_boxes * sizeof(box_str);

	// allocate boxes
	sprintf(fpath, "%s/box.mem", folder);
	if ((fp = fopen(fpath, "w+")) == NULL) {
		fprintf(stderr, "Cannot open %s\n", fpath);
		abort();
		exit(EXIT_FAILURE);
	}
	if (ftruncate(fileno(fp), size) != 0) {
		fprintf(stderr, "error: cannot truncate %s %ld\n", fpath, size);
		perror("ftruncate");
		exit(EXIT_FAILURE);
	}

	box_cpu = (box_str *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fileno(fp), 0);
	if (box_cpu == NULL) {
		fprintf(stderr, "ERROR: Cannot mmap box_cpu\n");
		exit(EXIT_FAILURE);
	}

	// initialize number of home boxes
	fill_box_data(box_cpu, boxsize);

	fclose(fp);	
}

static void
make_mem_file(const char *fname, int n_elems)
{
	char	fpath[512];
	FILE	*fp;
	int	i;

	snprintf(fpath, 512, "%s/%s", folder, fname);
	if ((fp = fopen(fpath, "w+")) == NULL) {
		fprintf(stderr, "Cannot open for creating empty file: %s\n", fpath);
		abort();
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < n_elems; i++) {
		fp_t	val;

		val = (rand() % 10 + 1) / 10.0;
		fwrite(&val, sizeof(fp_t), 1, fp);
	}
	fclose(fp);
}

static void
confer_save(FILE *fp, const char *fpath, void *ctx)
{
	int	boxsize = *(int *)ctx;
	fprintf(fp, "%u", boxsize);
}

int 
main(int argc, char *argv [])
{
	// counters
	long i, j, k, l, m, n;

	// system memory
	par_str	par_cpu;
	dim_str	dim_cpu;
	FILE	*f;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <boxsize> <folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);

	dim_cpu.cur_arg = 1;
	dim_cpu.boxes1d_arg = atoi(argv[1]);

	if (dim_cpu.boxes1d_arg < 0) {
		fprintf(stderr, "ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
		abort();
		exit(EXIT_FAILURE);
	}

	folder = argv[2];

	cuio_init(CUIO_TYPE_GENERATOR, folder);
	cuio_save_conf(confer_save, &dim_cpu.boxes1d_arg);

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

	make_box_file(dim_cpu.number_boxes, dim_cpu.boxes1d_arg);

	srand(1234);

	// input (distances)
	make_mem_file("rv.mem", dim_cpu.space_elem * 4);
	// input (charge)
	make_mem_file("qv.mem", dim_cpu.space_elem);

	return 0;
}
