#ifndef _UVM_UXU_ANAL_H
#define _UVM_UXU_ANAL_H

#include "uvm_linux.h"

void uXuA_printk(char catchar, const char *fmt, ...);

void uXuA_printk_addr(unsigned long long start_addr, char *hex_page_mask, unsigned long length);

#endif
