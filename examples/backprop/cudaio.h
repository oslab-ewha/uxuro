#ifndef _CUDA_IO_H_
#define _CUDA_IO_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	CUIO_MODE_NONE,
	CUIO_MODE_READONLY,
	CUIO_MODE_READWRITE,
	CUIO_MODE_WRITEONLY
} cuio_mode_t;

void *cuio_alloc_data(size_t length);
void cuio_free_data(void *data);

float *cuio_load_floats(const char *fname, size_t count, cuio_mode_t mode);
void cuio_unload_floats(const char *fname, size_t count, float *data, cuio_mode_t mode);
	
#ifdef __cplusplus
}
#endif

#endif
