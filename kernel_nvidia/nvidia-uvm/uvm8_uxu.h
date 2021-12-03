#ifndef __UVM8_UXU_H__
#define __UVM8_UXU_H__

#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"

// Flags for each mapping
#define UVM_UXU_FLAG_READ        0x01
#define UVM_UXU_FLAG_WRITE       0x02
#define UVM_UXU_FLAG_CREATE      0x04
#define UVM_UXU_FLAG_DONTTRASH   0x08
#define UVM_UXU_FLAG_VOLATILE    0x10
/* Not used. UXU always uses host buffer(page cache). */
#define UVM_UXU_FLAG_USEHOSTBUF  0x20

NV_STATUS uxu_init(void);
void uxu_exit(void);

NV_STATUS uvm_uxu_unregister_va_range(uvm_va_range_t *va_range);

void uxu_try_load_block(uvm_va_block_t *block,
			uvm_va_block_retry_t *block_retry,
			uvm_service_block_context_t *service_context,
			uvm_processor_id_t processor_id);

struct page *uxu_get_page(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero);

void uvm_uxu_set_page_dirty(struct page *page);

void stop_pagecache_reducer(uvm_va_space_t *va_space);

NV_STATUS
uxu_va_block_make_resident(uvm_va_block_t *va_block,
			   uvm_va_block_retry_t *va_block_retry,
			   uvm_va_block_context_t *va_block_context,
			   uvm_processor_id_t dest_id,
			   uvm_va_block_region_t region,
			   const uvm_page_mask_t *page_mask,
			   const uvm_page_mask_t *prefetch_page_mask,
			   uvm_make_resident_cause_t cause);

/**
 * Is this va_range managed by uxu driver?
 *
 * @param va_range: va_range to be examined.
 * @return: true if this va_range is managed by uxu driver, false otherwise.
 */
static inline bool
uvm_is_uxu_range(uvm_va_range_t *range)
{
	return range->node.uxu_rtn.filp != NULL;
}

static inline bool
uvm_is_uxu_block(uvm_va_block_t *block)
{
	return uvm_is_uxu_range(block->va_range);
}

static inline bool
uxu_check_flag(uvm_va_block_t *block, unsigned short flag)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn = &block->va_range->node.uxu_rtn;

	if (uxu_rtn->flags & flag)
		return true;
	return false;
}

#define uxu_is_write_block(block)	uxu_check_flag(block, UVM_UXU_FLAG_WRITE)
#define uxu_is_volatile_block(block)	uxu_check_flag(block, UVM_UXU_FLAG_VOLATILE)

#endif
