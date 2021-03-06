#include "unit_test.h"

static void
unit_test_write(unsigned long size)
{
	do_map_for_write(size);

	RUN_WRITE_KERNEL(size);

	do_unmap_for_write();

	drop_caches();

	if (!check_tmpfile(size))
		FAIL("unexpected written file");

	cleanup();

	pass("write only(%dKB)", size / 1024);
}

static void
unit_test_check_write(unsigned long size)
{
	do_map_for_write(size);

	RUN_WRITE_KERNEL(size);

	if (!check_buf(size))
		FAIL("invalid uxu buffer written by GPU");

	do_unmap_for_write();

	drop_caches();

	if (!check_tmpfile(size))
		FAIL("invalid written file");

	cleanup();

	pass("write checked(%dMB)", size / 1024 / 1024);
}

static void
unit_test_host_write_only(unsigned long size)
{
	drop_cache_before_map = TRUE;
	do_map_for_write(size);
	drop_cache_before_map = FALSE;

	write_reversed(size);

	if (!check_tmpfile(size))
		FAIL("unexpected kernel write behavior");

	do_unmap_for_write();

	cleanup();

	pass("fault by host write (%dKB)", size / 1024);
}

static void
unit_test_multiple_writes(unsigned long size)
{
	do_map_for_write(size);

	RUN_WRITE_KERNEL(size);

	if (!check_buf(size))
		FAIL("invalid uxu buffer written by GPU");

	write_reversed(size);
	write_reversed(size);

	RUN_WRITE_KERNEL(size);

	if (!check_buf(size))
		FAIL("invalid uxu buffer written by GPU");

	do_unmap_for_write();

	drop_caches();

	if (!check_tmpfile(size))
		FAIL("unexpected kernel write behavior");

	cleanup();

	pass("multiple writes (%dMB)", size / 1024 / 1024);
}

int
main(int argc, char *argv[])
{
	unit_test_write(8 * KB);

	unit_test_check_write(8 * MB);

	unit_test_host_write_only(8 * KB);

	unit_test_multiple_writes(18 * MB);

	return 0;
}
