#ifndef _CUDA_IO_H_
#define _CUDA_IO_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUIO_FLOATS_H(floats)	((float *)(floats).ptr_h)
#define CUIO_FLOATS_D(floats)	((float *)(floats).ptr_d)
#define CUIO_FLOATS_ITEM(floats, i)	((float *)(floats).ptr_h)[i]

typedef enum {
	CUIO_TYPE_NONE,
	CUIO_TYPE_HOST,
	CUIO_TYPE_UVM,
	CUIO_TYPE_HOSTREG,
	CUIO_TYPE_DRAGON
} cuio_type_t;

typedef enum {
	CUIO_MODE_READONLY,
	CUIO_MODE_READWRITE,
	CUIO_MODE_WRITEONLY
} cuio_mode_t;

typedef struct {
	void	*ptr_h, *ptr_d;
	size_t	size;
	cuio_type_t	type;
} cuio_ptr_t;

typedef void (*cuio_confer_t)(FILE *fp, const char *fpath, void *ctx);

void cuio_init(cuio_type_t type, const char *folder, int create_folder);

cuio_ptr_t cuio_alloc_mem(size_t len);
void cuio_free_mem(cuio_ptr_t *pptr);
void cuio_memcpy_h2d(cuio_ptr_t *pptr);
void cuio_memcpy_d2h(cuio_ptr_t *pptr);
void cuio_memset_d(cuio_ptr_t *pptr, int val);

void cuio_load_conf(cuio_confer_t func, void *ctx);
void cuio_save_conf(cuio_confer_t func, void *ctx);

cuio_ptr_t cuio_load_floats(const char *fname, size_t count, cuio_mode_t mode);
void cuio_unload_floats(const char *fname, cuio_ptr_t *pptr);

#ifdef __cplusplus
}
#endif

#endif
