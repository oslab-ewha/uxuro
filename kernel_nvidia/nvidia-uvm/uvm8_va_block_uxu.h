#ifndef __UVM8_VA_BLOCK_UXU_H__
#define __UVM8_VA_BLOCK_UXU_H__

#include "uvm8_uxu.h"

enum _block_transfer_mode_internal;

static NV_STATUS block_populate_page_cpu(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero);

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
_uxu_block_populate_page_cpu(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero)
{
	if (uxu_is_uxu_block(block))
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
__uxu_copy_resident_pages_mask(uvm_va_block_t *block,
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

	if (!uvm_uxu_is_managed(block->va_range) ||
	    (cause != UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE && cause != UVM_MAKE_RESIDENT_CAUSE_UXU) ||
	    (cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE && UVM_ID_IS_CPU(dst_id) && uvm_uxu_need_to_evict_from_gpu(block)) ||
	    (cause == UVM_MAKE_RESIDENT_CAUSE_UXU && UVM_ID_IS_GPU(dst_id))) {
		return block_copy_resident_pages_mask(block,
						      block_context,
						      dst_id,
						      src_processor_mask,
						      region,
						      page_mask,
						      prefetch_page_mask,
						      transfer_mode,
						      max_pages_to_copy,
						      migrated_pages,
						      copied_pages_out,
						      tracker_out);
	}

	return NV_OK;
}

/* Ugly */
#define UXU_WRITE_PROLOG()	bool do_uxu_write = false

#define UXU_DO_WRITE()		\
	do {			\
		if (uvm_uxu_is_managed(va_range) &&		\
		    uvm_uxu_need_to_evict_from_gpu(va_block) &&	\
		    (cause == UVM_MAKE_RESIDENT_CAUSE_EVICTION || (cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE && UVM_ID_IS_CPU(dest_id)))) {	\
			uvm_uxu_range_tree_node_t *uxu_rtn = &va_block->va_range->node.uxu_rtn;	\
												\
			if (!va_block->is_dirty && (uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE)) {	\
				uvm_uxu_block_mark_recent_in_buffer(va_block);			\
			}									\
		        else {									\
				uvm_uxu_write_begin(va_block, cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);	\
				do_uxu_write = true;								\
			}											\
		}												\
	} while (0)

#define UXU_WRITE_EPILOG()	\
	do {			\
		if (do_uxu_write) {		\
			status = uvm_tracker_wait(&va_block->tracker);	\
			uvm_uxu_write_end(va_block, cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);	\
		        if (status != NV_OK)	\
				return status;	\
		}				\
	} while (0)

#endif
