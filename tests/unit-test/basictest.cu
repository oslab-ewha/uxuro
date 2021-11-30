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
unit_test_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);

	RUN_KERNEL((kernel_read<<<8, 32>>>(buf_uxu, size, d_read_result)));

	do_unmap_for_read();

	if (h_read_Result != 0)
		FAIL("unexpected kernel read behavior");

	cleanup();

	pass("%dKB read", size / 1024);
}

static void
unit_test_cached_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);
	read_all(size);

	RUN_KERNEL((kernel_read<<<8, 32>>>(buf_uxu, size, d_read_result)));

	do_unmap_for_read();

	if (h_read_Result != 0)
		FAIL("kernel failure");

	cleanup();

	pass("%dKB read", size / 1024);
}

static void
unit_test_write(unsigned long size)
{
	do_map_for_write(size);

	RUN_KERNEL((kernel_write<<<8, 32>>>(buf_uxu, size)));

	do_unmap_for_write();

	if (!check_tmpfile(size))
		FAIL("unexpected kernel write behavior");

	cleanup();

	pass("%dKB write", size / 1024);
}

static void
unit_test_write_read(unsigned long size)
{
	do_map_for_write(size);

	RUN_KERNEL((kernel_write<<<8, 32>>>(buf_uxu, size)));

	do_unmap_for_write();
	do_map_for_read(size);

	prepare_d_result();

	RUN_KERNEL((kernel_read<<<8, 32>>>(buf_uxu, size, d_read_result)));

	do_unmap_for_read();
	if (h_read_Result != 0)
		FAIL("unexpected kernel read behavior");

	cleanup();

	pass("%dKB write & read", size / 1024);
}

#define MB	(1024*1024)

int
main(int argc, char *argv[])
{
	unit_test_read(4096);
	unit_test_read(8192);
	unit_test_read(8000);
	unit_test_read(8000 * 4096);

	unit_test_cached_read(5 * MB);

	unit_test_write(8192);
	unit_test_write(18 * MB);

	unit_test_write_read(20 * MB);

	return 0;
}
