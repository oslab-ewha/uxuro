#include "config.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <glib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "dragon.h"

#define PSF_DIR		"/proc/self/fd"
#define NVIDIA_UVM_PATH	"/dev/nvidia-uvm"
#define DRAGON_IOCTL_INIT			1000
#define DRAGON_IOCTL_MAP			1001
#define DRAGON_IOCTL_TRASH_NRBLOCKS		1002
#define DRAGON_IOCTL_TRASH_RESERVED_NRPAGES	1003
#define DRAGON_IOCTL_REMAP			1004

#define MIN_SIZE			((size_t)1 << 21)
#define DEFAULT_TRASH_NR_BLOCKS		32
#define DEFAULT_TRASH_NR_RESERVED_PAGES	(((unsigned long)1 << 21) * 4)

#define DRAGON_ENVNAME_ENABLE_READ_CACHE	"DRAGON_ENABLE_READ_CACHE"
#define DRAGON_ENVNAME_ENABLE_LAZY_WRITE	"DRAGON_ENABLE_LAZY_WRITE"
#define DRAGON_ENVNAME_ENABLE_AIO_READ		"DRAGON_ENABLE_AIO_READ"
#define DRAGON_ENVNAME_ENABLE_AIO_WRITE		"DRAGON_ENABLE_AIO_WRITE"
#define DRAGON_ENVNAME_READAHEAD_TYPE		"DRAGON_READAHEAD_TYPE"
#define DRAGON_ENVNAME_NR_RESERVED_PAGES	"DRAGON_NR_RESERVED_PAGES"

#define DRAGON_INIT_FLAG_ENABLE_READ_CACHE	0x01
#define DRAGON_INIT_FLAG_ENABLE_LAZY_WRITE	0x02
#define DRAGON_INIT_FLAG_ENABLE_AIO_READ	0x04
#define DRAGON_INIT_FLAG_ENABLE_AIO_WRITE	0x08

static int	fadvice = -1;
static int	fd_uvm = -1;
static int	initialized;

static int	minsize = MIN_SIZE;
static int	disabled_dragon = 0;

static GHashTable	*addr_map;

typedef struct {
	unsigned long trash_nr_blocks;
	unsigned long trash_reserved_nr_pages;
	unsigned short flags;
	unsigned int status;
} dragon_ioctl_init_t;

typedef struct {
	int backing_fd;
	void *uvm_addr;
	size_t size;
	unsigned short flags;
	unsigned int status;
} dragon_ioctl_map_t;

static int
open_uvm_dev(void)
{
	int	fd;

	fd = open(NVIDIA_UVM_PATH, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "failed to open: %s\n", NVIDIA_UVM_PATH);
	}
	return fd;
}

static dragonError_t
try_to_init_uvm(void)
{
	void	*addr;
	cudaError_t	error;

	/* dummy cudaMallocManaged() will initialize uvm */
	error = cudaMallocManaged(&addr, 1, cudaMemAttachGlobal);
	if (error != cudaSuccess) {
		fprintf(stderr, "failed to cudaMallocManaged: %s %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
		return D_ERR_UVM;
	}
	cudaFree(addr);
	return D_OK;
}

static dragonError_t
setup_fd_uvm(void)
{
	DIR	*dir;
	struct dirent	*ent;
	dragonError_t	err = D_ERR_FILE;

	dir = opendir(PSF_DIR);
	if (dir == NULL)
		return D_ERR_FILE;

	while ((ent = readdir(dir)) != NULL) {
		char	*psf_path, *psf_realpath;

		if (ent->d_type != DT_LNK)
			continue;

		asprintf(&psf_path, "%s/%s", PSF_DIR, ent->d_name);
		psf_realpath = realpath(psf_path, NULL);
		free(psf_path);
		if (psf_realpath == NULL)
			continue;
		if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
			fd_uvm = atoi(ent->d_name);
		free(psf_realpath);
		if (fd_uvm >= 0) {
			err = D_OK;
			break;
		}
	}
	closedir(dir);

	return err;
}

static dragonError_t
init_module(void)
{
	dragon_ioctl_init_t	request;
	long	nr_pages;
	char	*env_val;
	char	*endptr;
	int	status;
	dragonError_t	err;

	addr_map = g_hash_table_new(NULL, NULL);

	if (secure_getenv("NO_DRAGON")) {
		disabled_dragon = 1;
		initialized = 1;
		return D_OK;
	}

	if ((err = try_to_init_uvm()) != D_OK) {
		fprintf(stderr, "failed to initialize dragon: %d\n", err);
		return D_ERR_UVM;
	}

	if ((err = setup_fd_uvm()) != D_OK) {
		fprintf(stderr, "failed to get uvm fd: %d\n", err);
		return D_ERR_UVM;
	}

	env_val = secure_getenv("DRAGON_MINSIZE");
	if (env_val) {
		sscanf(env_val, "%u", &minsize);
	}

	request.trash_nr_blocks = DEFAULT_TRASH_NR_BLOCKS;
	request.trash_reserved_nr_pages = DEFAULT_TRASH_NR_RESERVED_PAGES;
	request.flags = 0;

	env_val = secure_getenv(DRAGON_ENVNAME_NR_RESERVED_PAGES);
	if (env_val && (nr_pages = strtol(env_val, &endptr, 10)) >= 0) {
		if (*env_val != '\0' && *endptr == '\0')
			request.trash_reserved_nr_pages = (unsigned long)nr_pages;
	}
	env_val = secure_getenv(DRAGON_ENVNAME_ENABLE_READ_CACHE);
	if (!(env_val && strncasecmp(env_val, "no", 2) == 0))
		request.flags |= DRAGON_INIT_FLAG_ENABLE_READ_CACHE;

	env_val = secure_getenv(DRAGON_ENVNAME_ENABLE_LAZY_WRITE);
	if (!(env_val && strncasecmp(env_val, "no", 2) == 0))
		request.flags |= DRAGON_INIT_FLAG_ENABLE_LAZY_WRITE;

	env_val = secure_getenv(DRAGON_ENVNAME_ENABLE_AIO_READ);
	if (!(env_val && strncasecmp(env_val, "no", 2) == 0))
        request.flags |= DRAGON_INIT_FLAG_ENABLE_AIO_READ;

	env_val = secure_getenv(DRAGON_ENVNAME_ENABLE_AIO_WRITE);
	if (!(env_val && strncasecmp(env_val, "no", 2) == 0))
		request.flags |= DRAGON_INIT_FLAG_ENABLE_AIO_WRITE;

	env_val = secure_getenv(DRAGON_ENVNAME_READAHEAD_TYPE);
	if (env_val && strncasecmp(env_val, "agg", 3) == 0) {
		fadvice = POSIX_FADV_SEQUENTIAL;
		fprintf(stderr, "Aggressive read ahead is enabled.\n");
	}
	else if (env_val && strncasecmp(env_val, "dis", 3) == 0) {
		fadvice = POSIX_FADV_RANDOM;
		fprintf(stderr, "Read ahead is disabled.\n");
	}
	else
		fadvice = POSIX_FADV_NORMAL;

	if ((status = ioctl(fd_uvm, DRAGON_IOCTL_INIT, &request)) != 0) {
		fprintf(stderr, "ioctl init error: %d\n", status);
		close(fd_uvm);
		err = D_ERR_IOCTL;
	}
	else {
		initialized = 1;
	}

	return err;
}

#define BUFSIZE	(1024 * 4)

static dragonError_t
fillup_from_file(dragon_ioctl_map_t *request)
{
	size_t	size;
	unsigned char	*addr;

	if (!(request->flags & D_F_READ))
		return D_OK;

	addr = (unsigned char *)request->uvm_addr;
	size = request->size;
	while (size > 0) {
		char buf[BUFSIZE];
		int nread = BUFSIZE;

		if (nread > size)
			nread = size;
		nread = read(request->backing_fd, buf, nread);
		if (nread == 0)
			break;
		memcpy(addr, buf, nread);
		addr += nread;
		size -= nread;
	}

	return D_OK;
}

static void
flush_to_file(dragon_ioctl_map_t *request)
{
	size_t	size;
	unsigned char	*addr;

	if (!(request->flags & D_F_WRITE))
		return;

	addr = (unsigned char *)request->uvm_addr;
	size = request->size;
	while (size > 0) {
		char	buf[BUFSIZE];
		int	nwrite = BUFSIZE;

		if (nwrite > size)
			nwrite = size;

		memcpy(buf, addr, nwrite);
		nwrite = write(request->backing_fd, buf, nwrite);
		if (nwrite == 0)
			break;
		addr += nwrite;
		size -= nwrite;
	}
}

static dragonError_t
do_dragon_map(dragon_ioctl_map_t *request)
{
	int	status;
	dragonError_t	err = D_OK;

	if ((request->flags & D_F_READ) && !(request->flags & D_F_VOLATILE)) {
		if ((status = posix_fadvise(request->backing_fd, 0, 0, fadvice)) != 0)
			fprintf(stderr, "fadvise error: %d\n", status);
		if ((fadvice == POSIX_FADV_SEQUENTIAL) && readahead(request->backing_fd, 0, request->size) != 0)
			fprintf(stderr, "readahead error.\n");
	}

	if ((status = ioctl(fd_uvm, DRAGON_IOCTL_MAP, request)) != 0) {
		fprintf(stderr, "ioctl error: %d\n", status);
		err = D_ERR_IOCTL;
	}

	return err;
}

static void
free_request(dragon_ioctl_map_t *request)
{
	close(request->backing_fd);
	free(request);
}

dragonError_t
dragon_map(const char *filename, size_t size, unsigned short flags, void **paddr)
{
	int	f_flags = 0;
	int	f_fd;
	dragon_ioctl_map_t	*request;
	cudaError_t	error;
	int		ret = D_OK;

	if (!initialized) {
		ret = init_module();
		if (ret != D_OK)
			return ret;
	}

	if ((request = (dragon_ioctl_map_t *)calloc(1, sizeof(dragon_ioctl_map_t))) == NULL) {
		fprintf(stderr, "Cannot calloc dragon_ioctl_map_t\n");
		return D_ERR_MEM;
	}

	f_flags = O_RDWR | O_LARGEFILE;
	if (flags & D_F_CREATE) {
		f_fd = creat(filename, S_IRUSR | S_IWUSR);
		if (f_fd >= 0)
			close(f_fd);
	}

	if ((f_fd = open(filename, f_flags)) < 0) {
		fprintf(stderr, "Cannot open the file %s\n", filename);
		free(request);
		return D_ERR_FILE;
	}

	if ((flags & D_F_CREATE) && ftruncate(f_fd, size) != 0) {
		fprintf(stderr, "Cannot truncate the file %s\n", filename);
		close(f_fd);
		free(request);
		return D_ERR_FILE;
	}

	error = cudaMallocManaged(&request->uvm_addr, size, cudaMemAttachGlobal);
	if (error != cudaSuccess) {
		fprintf(stderr, "failed to cudaMallocManaged: %s %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
		close(f_fd);
		free(request);
		return D_ERR_UVM;
	}

	request->backing_fd = f_fd;
	request->size = size;
	request->flags = flags;

	if (disabled_dragon)
		ret = fillup_from_file(request);
	else
		ret = do_dragon_map(request);

	if (ret == D_OK) {
		*paddr = request->uvm_addr;
		g_hash_table_insert(addr_map, request->uvm_addr, request);
	}
	else
		free_request(request);

	return ret;
}

dragonError_t
dragon_remap(void *addr, unsigned short flags)
{
	int	status;
	dragon_ioctl_map_t	*request = g_hash_table_lookup(addr_map, addr);
	int	fd;
	dragonError_t	err = D_OK;

	if (request == NULL) {
		fprintf(stderr, "%p is not mapped via dragon_map\n", addr);
		return D_ERR_INTVAL;
	}

	fd = open_uvm_dev();
	if (fd < 0)
		return D_ERR_UVM;

	cudaDeviceSynchronize();

	request->flags = flags;

	if ((status = ioctl(fd, DRAGON_IOCTL_REMAP, request)) != 0) {
		fprintf(stderr, "ioctl error: %d\n", status);
		err = D_ERR_IOCTL;
	}

	close(fd);

	return err;
}

dragonError_t
dragon_trash_set_num_blocks(unsigned long nrblocks)
{
	return D_ERR_NOT_IMPLEMENTED;
}

dragonError_t
dragon_trash_set_num_reserved_sys_cache_pages(unsigned long nrpages)
{
	return D_ERR_NOT_IMPLEMENTED;
}

dragonError_t
dragon_flush(void *addr)
{
	return D_ERR_NOT_IMPLEMENTED;
}

dragonError_t
dragon_unmap(void *addr)
{
	dragon_ioctl_map_t	*request = g_hash_table_lookup(addr_map, addr);

	if (request == NULL) {
		fprintf(stderr, "%p is not mapped via dragon_map\n", addr);
		return D_ERR_INTVAL;
	}

	if (disabled_dragon) {
		flush_to_file(request);
	}
	else {
		if ((request->flags & D_F_WRITE) && !(request->flags & D_F_VOLATILE))
			fsync(request->backing_fd);
	}

	cudaFree(request->uvm_addr);
	g_hash_table_remove(addr_map, addr);
	free_request(request);

	return D_OK;
}
