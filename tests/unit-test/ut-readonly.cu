#include "unit_test.h"

static void
unit_test_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);

	RUN_READ_KERNEL(size);

	do_unmap_for_read();

	cleanup();

	pass("readonly (%dKB)", size / 1024);
}

static void
unit_test_cached_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);
	read_all(size);

	RUN_READ_KERNEL(size);

	do_unmap_for_read();

	cleanup();

	pass("cached readonly(%dKB)", size / 1024);
}

static void
unit_test_multi_read(unsigned long size)
{
	prepare_d_result();
	do_map_for_read(size);

	RUN_READ_KERNEL(size);

	check_read_result();

	if (!check_tmpfile(size))
		FAIL("wrong data returned from GPU");

	write_reversed(size);

	RUN_READ_KERNEL(size);

	if (!check_tmpfile(size))
		FAIL("wrong reversed data returned from GPU");

	do_unmap_for_read();

	cleanup();

	pass("multiple reads(%dMB)", size / 1024 / 1024);
}

int
main(int argc, char *argv[])
{
	unit_test_read(4 * KB);
	unit_test_read(8 * KB);
	unit_test_read(8000);
	unit_test_read(8000 * 4096);

	unit_test_cached_read(5 * MB);

	unit_test_multi_read(5 * MB);

	return 0;
}
