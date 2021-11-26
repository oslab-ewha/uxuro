#ifndef _LIBUXU_H_
#define _LIBUXU_H_

#include <stdio.h>

/* Flags for uxu_map */
#define UXU_FLAGS_READ		0x01
#define UXU_FLAGS_WRITE		0x02
#define UXU_FLAGS_CREATE	0x04
#define UXU_FLAGS_DONTTRASH	0x08
#define UXU_FLAGS_VOLATILE	0x10
#define UXU_FLAGS_USEHOSTBUF	0x20

/* Errors */
typedef enum {
	UXU_OK = 0,
	UXU_ERR_FILE,
	UXU_ERR_IOCTL,
	UXU_ERR_UVM,
	UXU_ERR_INTVAL,
	UXU_ERR_MEM,
	UXU_ERR_NOT_IMPLEMENTED
} uxu_err_t;

#ifdef __cplusplus
extern "C"
{
#endif
	uxu_err_t uxu_map(const char *filename, size_t size, unsigned short flags, void **addr);
	uxu_err_t uxu_remap(void *addr, unsigned short flags);
	uxu_err_t uxu_trash_set_num_blocks(unsigned long nrblocks);
	uxu_err_t uxu_trash_set_num_reserved_sys_cache_pages(unsigned long nrpages);
	uxu_err_t uxu_flush(void *addr);
	uxu_err_t uxu_unmap(void *addr);
#ifdef __cplusplus
}
#endif

#endif

