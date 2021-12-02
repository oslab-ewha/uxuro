#include "unit_test.h"

static void
test_write_full(unsigned long size_evict, unsigned long size)
{
	do_map_for_write(size_evict);

	RUN_WRITE_KERNEL(size_evict);

	CUDA_CHECK(cudaMallocManaged((void **)&buf_uvm, size), "cudaMallocManaged failed");

	RUN_WRITE_MEM_KERNEL(buf_uvm, size);

	do_unmap_for_write();

	if (!check_tmpfile(size_evict))
		FAIL("unexpected evicted write");

	cleanup();

	pass("evicted write block");
}

int
main(int argc, char *argv[])
{
	test_write_full(8 * 1024, (unsigned long)(3.5 * GB));

	return 0;
}
