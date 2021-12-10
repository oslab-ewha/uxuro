#ifndef __UVM8_VA_BLOCK_UXU_H__
#define __UVM8_VA_BLOCK_UXU_H__

#include "uvm8_uxu.h"

enum _block_transfer_mode_internal;

static NV_STATUS block_populate_page_cpu(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero);

static NV_STATUS block_map_phys_cpu_page_on_gpus(uvm_va_block_t *block, uvm_page_index_t page_index, struct page *page);

static NV_STATUS block_copy_resident_pages_mask(uvm_va_block_t *block,
						uvm_va_block_context_t *block_context,
						uvm_processor_id_t dst_id,
						const uvm_processor_mask_t *src_processor_mask,
						uvm_va_block_region_t region,
						const uvm_page_mask_t *page_mask,
						const uvm_page_mask_t *prefetch_page_mask,
						int transfer_mode,
						NvU32 max_pages_to_copy,
						uvm_page_mask_t *migrated_pages,
						NvU32 *copied_pages_out,
						uvm_tracker_t *tracker_out);

static inline NV_STATUS
uxu_block_populate_page_cpu(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero)
{
	uvm_va_block_test_t	*block_test = uvm_va_block_get_test(block);
	struct page	*page;
	NV_STATUS	status;

	if (block->cpu.pages[page_index])
		return NV_OK;

	UVM_ASSERT(!uvm_page_mask_test(&block->cpu.resident, page_index));

	// Return out of memory error if the tests have requested it. As opposed to
	// other error injection settings, this one is persistent.
	if (block_test && block_test->inject_cpu_pages_allocation_error)
		return NV_ERR_NO_MEMORY;

	page = uxu_get_page(block, page_index, zero);
	if (page == NULL)
		return NV_ERR_NO_MEMORY;

	status = block_map_phys_cpu_page_on_gpus(block, page_index, page);
	if (status != NV_OK)
		goto error;

	block->cpu.pages[page_index] = page;
	return NV_OK;

error:
	__free_page(page);
	return status;
}

static inline NV_STATUS
uxubk_populate_page_cpu(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero)
{
	if (uvm_is_uxu_block(block))
		return uxu_block_populate_page_cpu(block, page_index, zero);
	else
		return block_populate_page_cpu(block, page_index, zero);
}

/*
 * TODO: copy is always enabled when cause == UVM_MAKE_RESIDENT_CAUSE_EVICTION.
 * This will drop performance but it's a simple workaround.
 * A kernel panic occurs when copy with UVM_UXU_FLAG_READ is skipped.
 */
static inline NV_STATUS
uxubk_copy_resident_pages_mask(uvm_va_block_t *block,
			       uvm_va_block_context_t *block_context,
			       uvm_processor_id_t dst_id,
			       const uvm_processor_mask_t *src_processor_mask,
			       uvm_va_block_region_t region,
			       const uvm_page_mask_t *page_mask,
			       const uvm_page_mask_t *prefetch_page_mask,
			       int transfer_mode,
			       NvU32 max_pages_to_copy,
			       uvm_page_mask_t *migrated_pages,
			       NvU32 *copied_pages_out,
			       uvm_tracker_t *tracker_out)
{
	uvm_make_resident_cause_t	cause = block_context->make_resident.cause;

	if (uvm_is_uxu_block(block) && cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE &&
	    (!UVM_ID_IS_CPU(dst_id) || !uxu_is_write_block(block)))
		return NV_OK;

	return block_copy_resident_pages_mask(block, block_context,
					      dst_id, src_processor_mask,
					      region, page_mask, prefetch_page_mask,
					      transfer_mode, max_pages_to_copy,
					      migrated_pages, copied_pages_out, tracker_out);
}

#endif
