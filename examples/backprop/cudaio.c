#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cuda.h>

#include "cuhelper.h"

#ifdef NVMGPU
#include "dragon.h"
#endif

#include "cudaio.h"

void *
cuio_alloc_data(size_t length)
{
	void	*data;

#if defined(CUDAMEMCPY) || defined(GENERATAOR)
	data = malloc(length);
	if (data == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(EXIT_FAILURE);
	}
#else	
	CUDA_CALL_SAFE(cudaMallocManaged(&data, length, cudaMemAttachGlobal));
#endif
	return data;
}

void
cuio_free_data(void *data)
{
#if defined(CUDAMEMCPY) || defined(GENERATAOR)
	free(data);
#else	
	CUDA_CALL_SAFE(cudaFree(data));
#endif
}

static void
read_file(const char *fpath, size_t length, void *buf)
{
	FILE	*fp;

	if ((fp = fopen(fpath, "rb")) == 0) {
		fprintf(stderr, "Cannot open file: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	if (fread(buf, length, 1, fp) != 1) {
		fprintf(stderr, "Cannot read: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	fclose(fp);
}

static void *
load_by_read(const char *fname, size_t length)
{
	float	*data;

	data = cuio_alloc_data(length);
	read_file(fname, length, data);
	return data;
}

static void
unload_by_write(const char *fpath, size_t length, void *buf)
{
	FILE	*fp;

	if ((fp = fopen(fpath, "wb")) == 0) {
		fprintf(stderr, "Cannot open: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	if (fwrite(buf, length, 1, fp) != 1) {
		fprintf(stderr, "Cannot write: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	fflush(fp);
	fsync(fileno(fp));
	fclose(fp);
}

#ifdef NVMGPU

static void *
mmap_by_dragon(const char *fpath, size_t length, int flag)
{
	void	*addr;

	if (dragon_map(fpath, length, flag, (void **)&addr) != D_OK) {
		fprintf(stderr, "Cannot dragon_map %s\n", fpath);
		exit(EXIT_FAILURE);
	}

	return addr;
}

static void
munmap_by_dragon(const char *fpath, void *addr)
{
	if (dragon_unmap(addr) != D_OK) {
		fprintf(stderr, "Cannot dragon_unmap: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
}

#endif

#ifdef HOSTREG

static void *
mmap_by_hostreg(const char *fpath, size_t length)
{
	void	*addr;
	int	fd;

	if ((fd = open(fpath, O_LARGEFILE | O_RDWR)) < 0) {
		fprintf(stderr, "Cannot open file: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	addr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
	if (addr == MAP_FAILED) {
		fprintf(stderr, "Cannot mmap: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	CUDA_CALL_SAFE(cudaHostRegister(addr, length, cudaHostRegisterDefault));

	close(fd);
	return addr;
}

static void
munmap_by_hostreg(const char *fpath, void *addr, size_t length)
{
	CUDA_CALL_SAFE(cudaHostUnregister(addr));

	if (msync(addr, length, MS_SYNC) != 0) {
		fprintf(stderr, "Cannot msync: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	if (munmap(addr, length) != 0) {
		fprintf(stderr, "Cannot munmap: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
}

#endif

float *
cuio_load_floats(const char *fpath, size_t count, cuio_mode_t mode)
{
	float *data;
	size_t	len = count * sizeof(float);

#ifdef NVMGPU
	int	flags = 0;

	switch (mode) {
	case CUIO_MODE_READONLY:
		flags = D_F_READ | D_F_DONTTRASH;
		break;
	case CUIO_MODE_READWRITE:
		flags = D_F_READ | D_F_WRITE | D_F_DONTTRASH;
		break;
	case CUIO_MODE_WRITEONLY:
		flags = D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE;
		break;
	default:
		return cuio_alloc_data(len);
	}
	return mmap_by_dragon(fpath, len, flags);
#elif defined(HOSTREG)
	switch (mode) {
	case CUIO_MODE_READONLY:
	case CUIO_MODE_READWRITE:
		return mmap_by_hostreg(fpath, len);
	default:
		return cuio_alloc_data(len);
	}
#else
	switch (mode) {
	case CUIO_MODE_READONLY:
	case CUIO_MODE_READWRITE:
		return load_by_read(fpath, len);
	default:
		return cuio_alloc_data(len);
	}
#endif
}

void
cuio_unload_floats(const char *fpath, size_t count, float *data, cuio_mode_t mode)
{
	size_t	len = count * sizeof(float);

#ifdef NVMGPU
	switch (mode) {
	case CUIO_MODE_NONE:
		cuio_free_data(data);
		break;
	default:
		munmap_by_dragon(fpath, data);
		break;
	}
#elif defined(HOSTREG)
	switch (mode) {
	case CUIO_MODE_READONLY:
	case CUIO_MODE_READWRITE:
		munmap_by_hostreg(fpath, data, len);
		break;
	case CUIO_MODE_WRITEONLY:
		unload_by_write(fpath, len, data);
		break;
	default:
		cuio_free_data(data);
		break;
	}
#else
	switch (mode) {
	case CUIO_MODE_READWRITE:
	case CUIO_MODE_WRITEONLY:
		return unload_by_write(fpath, len, data);
	default:
		cuio_free_data(data);
		break;
	}
#endif
}
