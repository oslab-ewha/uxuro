#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <dejagnu.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "libuxu.h"

#include "cuhelper.h"
#include "timer.h"

typedef int	BOOL;
#define TRUE	1
#define FALSE	0

#define FAIL(msg...)	\
	do {		\
		cleanup();	\
		fail(msg);	\
		exit(1);	\
	} while (0)

#define CUDA_CHECK(stmt, errmsg)		\
	do {					\
	        cudaError_t err = stmt;		\
	        if (err != cudaSuccess)	{	\
			FAIL("%s: %s", errmsg, cudaGetErrorString(err));	\
		}				\
	} while (0)

#define RUN_KERNEL(stmt)			\
	do {					\
	        stmt;				\
		CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");	\
	} while (0)

static __global__ void
kernel_read(uint8_t *g_buf, unsigned long size, int *pd_result)
{
	size_t	idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
	size_t	cntall = gridDim.x * blockDim.x;
	unsigned long	i;

	for (i = idx; i < size; i += cntall) {
		if (g_buf[i] != (uint8_t)i)
			*pd_result = -1;
	}
}

static __global__ void
kernel_write(uint8_t *g_buf, unsigned long size)
{
	size_t	idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
	size_t	cntall = gridDim.x * blockDim.x;
	unsigned long	i;

	for (i = idx; i < size; i += cntall)
		g_buf[i] = (uint8_t)i;
}

static char	fpath_tmpfile[1024];
static uint8_t	*g_buf;
static int	*d_read_result, h_read_Result;

static void
cleanup(void)
{
	if (d_read_result) {
		cudaFree(d_read_result);
		d_read_result = NULL;
	}
	if (*fpath_tmpfile) {
		unlink(fpath_tmpfile);
		fpath_tmpfile[0] = '\0';
	}	
}

static void
drop_caches(void)
{
	FILE	*fp;

	fp = fopen("/proc/sys/vm/drop_caches", "w");
	if (fp == NULL) {
		printf("failed to open drop_caches. Insufficient permission?\n");
		return;
	}
	fprintf(fp, "1\n");
	fclose(fp);
}

static BOOL
create_tmpfile_for_read(unsigned long size)
{
	FILE	*fp;
	int	fd;
	unsigned long	i;

	strcpy(fpath_tmpfile, ".basictest.read.XXXXXX");
	fd = mkstemp(fpath_tmpfile);
	if (fd < 0)
		return FALSE;
	fp = fdopen(fd, "w+");
	if (fp == NULL) {
		close(fd);
		return FALSE;
	}
	for (i = 0; i < size; i++) {
		uint8_t	byte = (uint8_t)i;
		fwrite(&byte, 1, 1, fp);
	}
	fclose(fp);
	drop_caches();
	return TRUE;
}

static BOOL
create_tmpfile_for_write(unsigned long size)
{
	int	fd;

	strcpy(fpath_tmpfile, ".basictest.write.XXXXXX");
	fd = mkstemp(fpath_tmpfile);
	if (fd < 0)
		return FALSE;
	close(fd);
	return TRUE;
}

static void
prepare_d_result(void)
{
	CUDA_CHECK(cudaMalloc(&d_read_result, sizeof(int)), "cudaMalloc failed");
	CUDA_CHECK(cudaMemcpy(d_read_result, &h_read_Result, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice failed");
}

static BOOL
check_tmpfile(unsigned long size)
{
	FILE	*fp;
	unsigned long	i;

	fp = fopen(fpath_tmpfile, "r");
	if (fp == NULL)
		return FALSE;

	for (i = 0; i < size; i++) {
		uint8_t	byte;

		fread(&byte, 1, 1, fp);
		if (byte != (uint8_t)i) {
			fclose(fp);
			return FALSE;
		}
	}
	fclose(fp);
	return TRUE;
}

static void
read_all(unsigned long size)
{
	FILE	*fp;
	char	buf[4096];
	unsigned long	i;

	fp = fopen(fpath_tmpfile, "r");
	if (fp == NULL) {
		FAIL("cannot open for read");
	}
	for (i = 0; i < size; i += 4096)
		fread(buf, 4096, 1, fp);
	fclose(fp);
}

static void
do_map_for_read(unsigned long size)
{
	int	err;

	if (fpath_tmpfile[0] == '\0' && !create_tmpfile_for_read(size))
		FAIL("cannot create file");

	if ((err = uxu_map(fpath_tmpfile, size, UXU_FLAGS_READ, (void **)&g_buf)) != UXU_OK)
		FAIL("failed to map for read: err: %d", err);
}

static void
do_map_for_write(unsigned long size)
{
	int	err;

	if (!create_tmpfile_for_write(size))
		FAIL("cannot create file");
	   
	if ((err = uxu_map(fpath_tmpfile, size, UXU_FLAGS_WRITE | UXU_FLAGS_CREATE, (void **)&g_buf)) != UXU_OK)
		FAIL("failed to map for write: err: %d", err);
}

static void
do_unmap_for_read(void)
{
	int	err;

	if ((err = uxu_unmap(g_buf)) != UXU_OK)
		FAIL("failed to unmap for read: err: %d", err);

	CUDA_CHECK(cudaMemcpy(&h_read_Result, d_read_result, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpyDeviceToHost failed");
}

static void
do_unmap_for_write(void)
{
	int	err;

	if ((err = uxu_unmap(g_buf)) != UXU_OK)
		FAIL("failed to unmap for write: err: %d", err);
}

static void
unit_test_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);

	RUN_KERNEL((kernel_read<<<8, 32>>>(g_buf, size, d_read_result)));

	do_unmap_for_read();

	if (h_read_Result != 0)
		FAIL("unexpected kernel read behavior");

	cleanup();

	pass("%dKB read", size / 1024);
}

static void
unit_test_cached_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);
	read_all(size);

	RUN_KERNEL((kernel_read<<<8, 32>>>(g_buf, size, d_read_result)));

	do_unmap_for_read();

	if (h_read_Result != 0)
		FAIL("kernel failure");

	cleanup();

	pass("%dKB read", size / 1024);
}

static void
unit_test_write(unsigned long size)
{
	do_map_for_write(size);

	RUN_KERNEL((kernel_write<<<8, 32>>>(g_buf, size)));

	do_unmap_for_write();

	if (!check_tmpfile(size))
		FAIL("unexpected kernel write behavior");

	cleanup();

	pass("%dKB write", size / 1024);
}

static void
unit_test_write_read(unsigned long size)
{
	do_map_for_write(size);

	RUN_KERNEL((kernel_write<<<8, 32>>>(g_buf, size)));

	do_unmap_for_write();
	do_map_for_read(size);

	prepare_d_result();

	RUN_KERNEL((kernel_read<<<8, 32>>>(g_buf, size, d_read_result)));

	do_unmap_for_read();
	if (h_read_Result != 0)
		FAIL("unexpected kernel read behavior");

	cleanup();

	pass("%dKB write & read", size / 1024);
}

#define MB	(1024*1024)

int
main(int argc, char *argv[])
{
	unit_test_read(4096);
	unit_test_read(8192);
	unit_test_read(8000);
	unit_test_read(8000 * 4096);

	unit_test_cached_read(5 * MB);

	unit_test_write(8192);
	unit_test_write(18 * MB);

	unit_test_write_read(20 * MB);

	return 0;
}
