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
#include "unit_test.h"

static void
test_write_full(unsigned long size)
{
	do_map_for_write(4096);

	RUN_KERNEL((kernel_write<<<8, 32>>>(buf_uxu, 4096)));

	CUDA_CHECK(cudaMallocManaged((void **)&buf_uvm, size), "cudaMallocManaged failed");

	RUN_KERNEL((kernel_write<<<8, 32>>>(buf_uvm, size)));

	do_unmap_for_write();

	if (!check_tmpfile(4096))
		FAIL("unexpected evicted write");

	cleanup();

	pass("evicted write block");
}

int
main(int argc, char *argv[])
{
	test_write_full((unsigned long)(3.5 * GB));

	return 0;
}
