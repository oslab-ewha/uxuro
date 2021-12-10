#include "unit_test.h"

static void
unit_test_write_read(unsigned long size)
{
	do_map_for_write(size);

	RUN_WRITE_KERNEL(size);

	do_unmap_for_write();
	do_map_for_read(size);

	prepare_d_result();

	RUN_READ_KERNEL(size);

	do_unmap_for_read();

	cleanup();

	pass("write & read (%dMB)", size / 1024 / 1024);
}

#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>

int
main(int argc, char *argv[])
{
	unit_test_write_read(20 * MB);

	return 0;
}
