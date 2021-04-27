#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff

#define GPU
#define THREADS 256
#define WIDTH 16  // shared memory width  
#define HEIGHT 16 // shared memory height

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define NUM_THREAD 4  //OpenMP threads

#include "bpnn.h"
#include "cuhelper.h"

/*** User-level functions ***/

#ifdef __cplusplus
extern "C" {
#endif

extern long	layer_size;
extern char	*folder;
	
#ifdef __cplusplus
}
#endif

void bpnn_save();
BPNN *bpnn_read();

void read_file(const char *fname, size_t length, void *buf);
void write_file(const char *fname, size_t length, void *buf);

#endif
