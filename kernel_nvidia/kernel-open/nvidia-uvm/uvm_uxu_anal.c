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

// convert hex string to binary string
void
uXuA_printk_addr(unsigned long long start_addr, char *hex_page_mask, unsigned long length)
{
    if (uvm_uXuA_printk == 0)
        return;

    char    bin_page_mask[4] = "0000";
    int hex_idx;
    int idx;
    int base;
    int offset;

    for (hex_idx = length - 2, base = 0; hex_idx >= 0; hex_idx--) { //start index is (length - 2), because last string is '\0'.
        switch (hex_page_mask[hex_idx]) {
            case '0':
                //strncpy(page_mask, "0000", 4);
                goto convert_addr;
            case '1':
                //strncpy(page_mask, "0001", 4);
                bin_page_mask[3] = '1';
                goto convert_addr;
            case '2':
                //strncpy(page_mask, "0010", 4);
                bin_page_mask[2] = '1';
                goto convert_addr;
            case '3':
                //strncpy(page_mask, "0011", 4);
                bin_page_mask[3] = '1';
                bin_page_mask[2] = '1';
                goto convert_addr;
            case '4':
                //strncpy(page_mask, "0100", 4);
                bin_page_mask[1] = '1';
                goto convert_addr;
            case '5':
                //strncpy(page_mask, "0101", 4);
                bin_page_mask[1] = '1';
                bin_page_mask[3] = '1';
                goto convert_addr;
            case '6':
                //strncpy(page_mask, "0110", 4);
                bin_page_mask[1] = '1';
                bin_page_mask[2] = '1';
                goto convert_addr;
            case '7':
                //strncpy(page_mask, "0111", 4);
                bin_page_mask[1] = '1';
                bin_page_mask[2] = '1';
                bin_page_mask[3] = '1';
                goto convert_addr;
            case '8':
                //strncpy(page_mask, "1000", 4);
                bin_page_mask[0] = '1';
                goto convert_addr;
            case '9':
                //strncpy(page_mask, "1001", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[3] = '1';
                goto convert_addr;
            case 'a':
                //strncpy(page_mask, "1010", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[2] = '1';
                goto convert_addr;
            case 'b':
                //strncpy(page_mask, "1011", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[2] = '1';
                bin_page_mask[3] = '1';
                goto convert_addr;
            case 'c':
                //strncpy(page_mask, "1100", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[1] = '1';
                goto convert_addr;
            case 'd':
                //strncpy(page_mask, "1101", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[1] = '1';
                bin_page_mask[3] = '1';
                goto convert_addr;
            case 'e':
                //strncpy(page_mask, "1110", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[1] = '1';
                bin_page_mask[2] = '1';
                goto convert_addr;
            case 'f':
                //strncpy(page_mask, "1111", 4);
                bin_page_mask[0] = '1';
                bin_page_mask[1] = '1';
                bin_page_mask[2] = '1';
                bin_page_mask[3] = '1';
                goto convert_addr;
            case ':':
                continue;
        }
        convert_addr:
        for (idx = 3, offset = 0; idx >= 0; idx--, offset++) {
            if(bin_page_mask[idx] == '1') {
                printk(KERN_CONT "%llx,", start_addr + (unsigned long long)((base * 4 + offset) * 4096));
                bin_page_mask[idx] = '0';
            }
        }
        base++;
    }
}