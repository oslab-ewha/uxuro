#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "benchmark.h"

int
mb_parse_count(const char *arg, const char *name)
{
	unsigned	count;

	if (sscanf(optarg, "%u", &count) != 1)
		ERROR("invalid number of %s: %s", arg, name);
	if (count == 0)
		ERROR("0 %s not allowed", name);
	return count;
}

int
mb_parse_size(const char *arg, const char *name)
{
	unsigned	size;
	char	unit;

	if (sscanf(optarg, "%u%c", &size, &unit) == 2) {
		if (unit == 'k')
			size *= 1024;
		else if (unit == 'm')
			size *= 1024 * 1024;
		else if (unit == 'g')
			size *= 1024 * 1024 * 1024;
		else
			ERROR("invalid size unit");
	}
	if (size == 0)
		ERROR("0 %s is not allowed", name);
	return size;
}

int
mb_parse_procid(const char *arg)
{
	unsigned	gpuid;

	if (strcmp(arg, "cpu") == 0)
		return cudaCpuDeviceId;

	if (sscanf(arg, "%u", &gpuid) != 1)
		ERROR("invalid argument: %s\n", arg);
	return gpuid;
}

char *
mb_get_sizestr(unsigned long num)
{
	char	buf[1024];

	if (num < 1024)
		snprintf(buf, 1024, "%lu", num);
	else {
		num /= 1024;
		if (num < 1024)
			snprintf(buf, 1024, "%luk", num);
		else {
			num /= 1024;
			if (num < 1024)
				snprintf(buf, 1024, "%lum", num);
			else {
				num /= 1024;
				snprintf(buf, 1024, "%lug", num);
			}
		}
	}
	return strdup(buf);
}
