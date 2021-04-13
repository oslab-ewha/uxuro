GLIB_CFLAGS ?= $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS ?= $(shell pkg-config --libs glib-2.0)
CUDA_DIR ?= /usr/local/cuda

INCLUDES = -I../include -I$(CUDA_DIR)/include
CFLAGS = -fPIC -Wall -O3 -m64 $(GLIB_CFLAGS)
LDFLAGS = -L$(CUDA_DIR)/lib64
LIBS = -lcudart $(GLIB_LIBS)
DEFS = -D_GNU_SOURCE

all: dragon.c ../include/dragon.h
	gcc -shared -o libdragon.so $(INCLUDES) $(CFLAGS) $(DEFS) $(LDFLAGS) dragon.c $(LIBS)
	mkdir -p ../lib64/
	mv libdragon.so ../lib64/

clean:
	rm ../lib64/*.so
