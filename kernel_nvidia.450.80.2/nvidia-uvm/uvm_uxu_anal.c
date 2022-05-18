#include "uvm_uxu_anal.h"

#include "uvm_common.h"

static int uvm_uXuA_printk = 1;

module_param(uvm_uXuA_printk, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_uXuA_printk, "Enable uxu analysis printk");

void
uXuA_printk(char catchar, const char *fmt, ...)
{
	char	buf[512];
	va_list	args;

	if (uvm_uXuA_printk == 0)
		return;

	va_start(args, fmt);
	vsnprintf(buf, 512, fmt, args);
	va_end(args);

	printk("uXuA%c:%s\n", catchar, buf);
}
