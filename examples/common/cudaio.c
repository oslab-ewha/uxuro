#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/stat.h>

#include <cuda.h>

#include "cuhelper.h"
#include "dragon.h"
#include "cudaio.h"

static const char	*folder_base;
static cuio_type_t	type = CUIO_TYPE_NONE;

static void
check_folder(int create_folder)
{
	if (access(folder_base, F_OK) == 0) {
		if (create_folder) {
			fprintf(stderr, "folder exist: %s\n", folder_base);
			fprintf(stderr, "You should provide a non-existent folder\n");
			exit(1);
		}
	}
	else if (create_folder) {
		if (mkdir(folder_base, 0700) < 0) {
			fprintf(stderr, "cannot make directory: %s\n", folder_base);
			exit(2);
		}
	}
	else {
		fprintf(stderr, "folder does not exist: %s\n", folder_base);
		exit(1);
	}
}

void
cuio_init(cuio_type_t _type, const char *folder, int create_folder)
{
	type = _type;

	if (type == CUIO_TYPE_NONE) {
		const char	*typestr = getenv("CUIO_TYPE");

		if (typestr == NULL)
			type = CUIO_TYPE_HOST;
		else if (strcmp(typestr, "NVMGPU") == 0)
			type = CUIO_TYPE_DRAGON;
		else if (strcmp(typestr, "UVM") == 0)
			type = CUIO_TYPE_UVM;
		else if (strcmp(typestr, "HOSTREG") == 0)
			type = CUIO_TYPE_HOSTREG;
		else
			type = CUIO_TYPE_HOST;
	}
	folder_base = strdup(folder);
	check_folder(create_folder);
}

cuio_ptr_t
cuio_alloc_mem(size_t len)
{
	cuio_ptr_t	ptr;

	if (type == CUIO_TYPE_HOST) {
		ptr.ptr_h = malloc(len);
		if (ptr.ptr_h == NULL) {
			fprintf(stderr, "out of memory\n");
			exit(EXIT_FAILURE);
		}
		CUDA_CALL_SAFE(cudaMalloc((void**)&ptr.ptr_d, len));
	}
	else {
		CUDA_CALL_SAFE(cudaMallocManaged(&ptr.ptr_h, len, cudaMemAttachGlobal));
		ptr.ptr_d = ptr.ptr_h;
	}
	ptr.type = type;
	ptr.size = len;
	return ptr;
}

void
cuio_free_mem(cuio_ptr_t *pptr)
{
	if (pptr->type == CUIO_TYPE_DRAGON || pptr->type == CUIO_TYPE_HOSTREG)
		return;
	if (pptr->ptr_d)
		CUDA_CALL_SAFE(cudaFree(pptr->ptr_d));
	if (pptr->type == CUIO_TYPE_HOST && pptr->ptr_h)
		free(pptr->ptr_h);
}

void
cuio_memcpy_h2d(cuio_ptr_t *pptr)
{
	if (pptr->type != CUIO_TYPE_HOST)
		return;
	CUDA_CALL_SAFE(cudaMemcpy(pptr->ptr_d, pptr->ptr_h, pptr->size, cudaMemcpyHostToDevice));
	CUDA_CALL_SAFE(cudaDeviceSynchronize());
}

void
cuio_memcpy_d2h(cuio_ptr_t *pptr)
{
	if (pptr->type != CUIO_TYPE_HOST)
		return;
	CUDA_CALL_SAFE(cudaMemcpy(pptr->ptr_h, pptr->ptr_d, pptr->size, cudaMemcpyDeviceToHost));
	CUDA_CALL_SAFE(cudaDeviceSynchronize());
}

static void
read_file(const char *fpath, size_t len, cuio_ptr_t ptr)
{
	FILE	*fp;

	if ((fp = fopen(fpath, "rb")) == 0) {
		fprintf(stderr, "Cannot open file: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	if (fread(ptr.ptr_h, len, 1, fp) != 1) {
		fprintf(stderr, "Cannot read: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	fclose(fp);
}

static cuio_ptr_t
load_by_read(const char *fpath, size_t len)
{
	cuio_ptr_t	ptr;

	ptr = cuio_alloc_mem(len);
	read_file(fpath, len, ptr);
	return ptr;
}

static void
unload_by_write(const char *fpath, cuio_ptr_t *pptr)
{
	FILE	*fp;

	if ((fp = fopen(fpath, "wb")) == 0) {
		fprintf(stderr, "Cannot open: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	if (fwrite(pptr->ptr_h, pptr->size, 1, fp) != 1) {
		fprintf(stderr, "Cannot write: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	fflush(fp);
	fsync(fileno(fp));
	fclose(fp);

	cuio_free_mem(pptr);
}

static cuio_ptr_t
mmap_by_dragon(const char *fpath, size_t len, cuio_mode_t mode)
{
	cuio_ptr_t	ptr;
	int	flags = D_F_READ;

	switch (mode) {
	case CUIO_MODE_READONLY:
		flags |= D_F_DONTTRASH;
		break;
	case CUIO_MODE_READWRITE:
		flags |= (D_F_WRITE | D_F_DONTTRASH);
		break;
	case CUIO_MODE_WRITEONLY:
		flags |= (D_F_WRITE | D_F_CREATE | D_F_VOLATILE);
		break;
	}
	if (dragon_map(fpath, len, flags, (void **)&ptr.ptr_h) != D_OK) {
		fprintf(stderr, "Cannot dragon_map %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	ptr.ptr_d = ptr.ptr_h;
	ptr.size = len;
	ptr.type = CUIO_TYPE_DRAGON;
	return ptr;
}

static void
munmap_by_dragon(const char *fpath, cuio_ptr_t *pptr)
{
	if (dragon_unmap(pptr->ptr_h) != D_OK) {
		fprintf(stderr, "Cannot dragon_unmap: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
}

static cuio_ptr_t
mmap_by_hostreg(const char *fpath, size_t len)
{
	cuio_ptr_t	ptr;
	int	fd;

	if ((fd = open(fpath, O_LARGEFILE | O_RDWR)) < 0) {
		fprintf(stderr, "Cannot open file: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	ptr.ptr_h = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
	close(fd);

	if (ptr.ptr_h == MAP_FAILED) {
		fprintf(stderr, "Cannot mmap: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	CUDA_CALL_SAFE(cudaHostRegister(ptr.ptr_h, len, cudaHostRegisterDefault));
	ptr.ptr_d = ptr.ptr_h;
	ptr.size = len;
	ptr.type = CUIO_TYPE_HOSTREG;
	return ptr;
}

static void
munmap_by_hostreg(const char *fpath, cuio_ptr_t *pptr)
{
	CUDA_CALL_SAFE(cudaHostUnregister(pptr->ptr_h));

	if (msync(pptr->ptr_h, pptr->size, MS_SYNC) != 0) {
		fprintf(stderr, "Cannot msync: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
	if (munmap(pptr->ptr_h, pptr->size) != 0) {
		fprintf(stderr, "Cannot munmap: %s\n", fpath);
		exit(EXIT_FAILURE);
	}
}

void
cuio_load_conf(cuio_confer_t confer, void *ctx)
{
	FILE	*fp;
	char	fpath[256];

	snprintf(fpath, 256, "%s/cuio.conf", folder_base);
	fp = fopen(fpath, "r");
	if (fp == NULL) {
		fprintf(stderr, "cannot open: %s\n", fpath);
		exit(2);
	}
	confer(fp, fpath, ctx);
	fclose(fp);
}

void
cuio_save_conf(cuio_confer_t confer, void *ctx)
{
	FILE	*fp;
	char	fpath[256];

	snprintf(fpath, 256, "%s/cuio.conf", folder_base);
	fp = fopen(fpath, "w");
	if (fp == NULL) {
		fprintf(stderr, "cannot open for write: %s\n", fpath);
		exit(2);
	}
	confer(fp, fpath, ctx);
	fclose(fp);	
}

cuio_ptr_t
cuio_load_floats(const char *fname, size_t count, cuio_mode_t mode)
{
	char	fpath[256];
	float	*data;
	size_t	len = count * sizeof(float);
	int	flags = 0;

	snprintf(fpath, 256, "%s/%s", folder_base, fname);

	switch (type) {
	case CUIO_TYPE_DRAGON:
		return mmap_by_dragon(fpath, len, mode);
	case CUIO_TYPE_HOSTREG:
		return mmap_by_hostreg(fpath, len);
	default:
		return load_by_read(fpath, len);
	}
}

void
cuio_unload_floats(const char *fname, cuio_ptr_t *pptr)
{
	char	fpath[256];

	snprintf(fpath, 256, "%s/%s", folder_base, fname);

	switch (pptr->type) {
	case CUIO_TYPE_DRAGON:
		munmap_by_dragon(fpath, pptr);
		break;
	case CUIO_TYPE_HOSTREG:
		munmap_by_hostreg(fname, pptr);
		break;
	default:
		unload_by_write(fpath, pptr);
		break;
	}
}
