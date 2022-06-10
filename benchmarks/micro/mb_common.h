#ifndef _MB_COMMON_H_
#define _MB_COMMON_H_

#include "benchmark.h"
#include "timer.h"

int mb_parse_count(const char *arg, const char *name);
int mb_parse_size(const char *arg, const char *name);
int mb_parse_procid(const char *arg);
char *mb_get_sizestr(unsigned long num);

#endif
