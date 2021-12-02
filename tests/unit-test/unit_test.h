#ifndef _UNIT_TEST_H_
#define _UNIT_TEST_H_

#include <unistd.h>
#include <dejagnu.h>

#include "libuxu.h"

#define KB	(1024)
#define MB	(1024*1024)
#define GB	(1024*1024*1024)

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

static char	fpath_tmpfile[1024];
static uint8_t	*buf_uxu, *buf_uvm;
static int	*d_read_result, h_read_Result;
static BOOL	is_ascending_order;

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

static __global__ void
kernel_read(uint8_t *buf_uxu, unsigned long size, int *pd_result, BOOL is_ascending_order)
{
	size_t	idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
	size_t	cntall = gridDim.x * blockDim.x;
	unsigned long	i;

	for (i = idx; i < size; i += cntall) {
		uint8_t	byte = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);
		if (buf_uxu[i] != byte)
			*pd_result = -1;
	}
}

static __global__ void
kernel_write(uint8_t *buf_uxu, unsigned long size, BOOL is_ascending_order)
{
	size_t	idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
	size_t	cntall = gridDim.x * blockDim.x;
	unsigned long	i;

	for (i = idx; i < size; i += cntall) {
		uint8_t	byte = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);
		buf_uxu[i] = byte;
	}
}

static inline void
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

static inline BOOL
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
		uint8_t	byte = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);
		fwrite(&byte, 1, 1, fp);
	}
	fclose(fp);
	drop_caches();
	return TRUE;
}

static inline BOOL
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

static inline void
prepare_d_result(void)
{
	CUDA_CHECK(cudaMalloc(&d_read_result, sizeof(int)), "cudaMalloc failed");
	CUDA_CHECK(cudaMemcpy(d_read_result, &h_read_Result, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice failed");
}

static inline BOOL
check_buf(unsigned long size)
{
	unsigned long	i;

	for (i = 0; i < size; i++) {
		uint8_t	checked = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);

		if (buf_uxu[i] != checked)
			return FALSE;
	}
	return TRUE;
}

static inline BOOL
check_tmpfile(unsigned long size)
{
	FILE	*fp;
	unsigned long	i;

	fp = fopen(fpath_tmpfile, "r");
	if (fp == NULL)
		return FALSE;

	for (i = 0; i < size; i++) {
		uint8_t	byte;
		uint8_t	checked = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);

		fread(&byte, 1, 1, fp);
		if (byte != checked) {
			fclose(fp);
			return FALSE;
		}
	}
	fclose(fp);
	return TRUE;
}

static inline void
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

static inline void
write_reversed(unsigned long size)
{
	unsigned long	i;

	is_ascending_order = !is_ascending_order;

	for (i = 0; i < size; i++) {
		uint8_t	byte = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);
		buf_uxu[i] = byte;
	}
}

static inline void
write_reversed_file(unsigned long size)
{
	FILE	*fp;
	unsigned long	i;

	is_ascending_order = !is_ascending_order;

	fp = fopen(fpath_tmpfile, "w");
	if (fp == NULL) {
		FAIL("cannot open for write");
	}
	for (i = 0; i < size; i++) {
		uint8_t	byte = is_ascending_order ? (uint8_t)i: (255 - (uint8_t)i);
		fwrite(&byte, 1, 1, fp);
	}
	fclose(fp);
}

static inline void
do_map_for_read(unsigned long size)
{
	int	err;

	if (fpath_tmpfile[0] == '\0' && !create_tmpfile_for_read(size))
		FAIL("cannot create file");

	if ((err = uxu_map(fpath_tmpfile, size, UXU_FLAGS_READ, (void **)&buf_uxu)) != UXU_OK)
		FAIL("failed to map for read: err: %d", err);
}

static inline void
do_map_for_write(unsigned long size)
{
	int	err;

	if (!create_tmpfile_for_write(size))
		FAIL("cannot create file");

	if ((err = uxu_map(fpath_tmpfile, size, UXU_FLAGS_WRITE | UXU_FLAGS_CREATE, (void **)&buf_uxu)) != UXU_OK)
		FAIL("failed to map for write: err: %d", err);
}

static void
check_read_result(void)
{
	CUDA_CHECK(cudaMemcpy(&h_read_Result, d_read_result, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpyDeviceToHost failed");
	if (h_read_Result != 0)
		FAIL("unsuccessful h_result: %d", h_read_Result);
}


static void
do_unmap_for_read(void)
{
	int	err;

	if ((err = uxu_unmap(buf_uxu)) != UXU_OK)
		FAIL("failed to unmap for read: err: %d", err);
	check_read_result();
}

static void
do_unmap_for_write(void)
{
	int	err;

	if ((err = uxu_unmap(buf_uxu)) != UXU_OK)
		FAIL("failed to unmap for write: err: %d", err);
}

#define RUN_READ_KERNEL(size)	do {	\
		RUN_KERNEL((kernel_read<<<8, 32>>>(buf_uxu, size, d_read_result, is_ascending_order))); \
	} while (0)

#define RUN_WRITE_MEM_KERNEL(buf, size)	do {				\
		RUN_KERNEL((kernel_write<<<8, 32>>>(buf, size, is_ascending_order))); \
	} while (0)

#define RUN_WRITE_KERNEL(size)	do {	\
		RUN_WRITE_MEM_KERNEL(buf_uxu, size);	\
	} while (0)

#endif
