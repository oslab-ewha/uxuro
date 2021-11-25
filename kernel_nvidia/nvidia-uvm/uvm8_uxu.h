#ifndef __UVM8_UXU_H__
#define __UVM8_UXU_H__

#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"

// Flags for each mapping
#define UVM_UXU_FLAG_READ        0x01
#define UVM_UXU_FLAG_WRITE       0x02
#define UVM_UXU_FLAG_DONTTRASH   0x08
#define UVM_UXU_FLAG_VOLATILE    0x10
#define UVM_UXU_FLAG_USEHOSTBUF  0x20

NV_STATUS uvm_uxu_initialize(uvm_va_space_t *va_space,
			     unsigned long trash_nr_blocks,
			     unsigned long trash_reserved_nr_pages,
			     unsigned short flags);
NV_STATUS uvm_uxu_register_file_va_space(uvm_va_space_t *va_space,
					 UVM_UXU_REGISTER_FILE_VA_SPACE_PARAMS *params);
NV_STATUS uvm_uxu_remap(uvm_va_space_t *va_space,
			UVM_UXU_REMAP_PARAMS *params);
NV_STATUS uvm_uxu_unregister_va_range(uvm_va_range_t *va_range);

NV_STATUS uvm_uxu_flush_host_block(uvm_va_space_t *va_space,
				   uvm_va_range_t *va_range,
				   uvm_va_block_t *va_block, bool is_evict,
				   const uvm_page_mask_t *page_mask);
NV_STATUS uvm_uxu_flush_block(uvm_va_block_t *va_block);
NV_STATUS uvm_uxu_flush(uvm_va_range_t *va_range);
NV_STATUS uvm_uxu_release_block(uvm_va_block_t *va_block);

NV_STATUS uvm_uxu_read_begin(uvm_va_block_t *va_block,
			     uvm_va_block_retry_t *block_retry,
			     uvm_service_block_context_t *service_context);
NV_STATUS uvm_uxu_read_end(uvm_va_block_t *va_block);

NV_STATUS uvm_uxu_write_begin(uvm_va_block_t *va_block, bool is_flush);
NV_STATUS uvm_uxu_write_end(uvm_va_block_t *va_block, bool is_flush);

NV_STATUS uvm_uxu_reduce_memory_consumption(uvm_va_space_t *va_space);

NV_STATUS uvm_uxu_prepare_block_for_hostbuf(uvm_va_block_t *va_block);

void uvm_uxu_set_page_dirty(struct page *page);

struct page *assign_pagecache(uvm_va_block_t * block, uvm_page_index_t page_index);

void stop_pagecache_reducer(uvm_va_space_t *va_space);

/**
 * Is this va_range managed by uxu driver?
 *
 * @param va_range: va_range to be examined.
 * @return: true if this va_range is managed by uxu driver, false otherwise.
 */
static inline bool
uvm_uxu_is_managed(uvm_va_range_t *va_range)
{
	return va_range->node.uxu_rtn.filp != NULL;
}

/**
 * Determine if we need to reclaim some blocks or not.
 *
 * @param uxu_va_space: the va_space information related to UXU.
 *
 * @return: true if we need to reclaim, false otherwise.
 */
static inline bool
uvm_uxu_has_to_reclaim_blocks(uvm_uxu_va_space_t *uxu_va_space)
{
	unsigned long	freeram = global_zone_page_state(NR_FREE_PAGES);
	unsigned long	pagecacheram = global_zone_page_state(NR_FILE_PAGES);
	return freeram + pagecacheram < uxu_va_space->trash_reserved_nr_pages;
}

static inline bool
uvm_uxu_block_file_dirty(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	return test_bit(bitmap_index, &uxu_rtn->is_file_dirty_bitmaps[list_index]);
}

static inline bool
uvm_uxu_need_to_copy_from_file(uvm_va_block_t *va_block,
			       uvm_processor_id_t
			       processor_id)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_block->va_range->node.uxu_rtn;

	if (!uvm_uxu_is_managed(va_block->va_range))
		return false;

	if (uvm_uxu_block_file_dirty(va_block))
		return true;

	return (!(uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE)	&&
		!((uxu_rtn->flags & UVM_UXU_FLAG_USEHOSTBUF) && va_block->uxu_use_uvm_buffer) &&
		((uxu_rtn->flags & UVM_UXU_FLAG_READ) || UVM_ID_IS_CPU(processor_id)));
}

static inline void
uvm_uxu_block_clear_has_data(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	clear_bit(bitmap_index, &uxu_rtn->has_data_bitmaps[list_index]);
}

static inline void
uvm_uxu_block_set_has_data(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	set_bit(bitmap_index, &uxu_rtn->has_data_bitmaps[list_index]);
}

static inline bool
uvm_uxu_block_has_data(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	return test_bit(bitmap_index, &uxu_rtn->has_data_bitmaps[list_index]);
}

static inline void
uvm_uxu_block_clear_file_dirty(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	size_t index = uvm_va_range_block_index(va_range, va_block->start);
	size_t list_index = index / BITS_PER_LONG;
	size_t bitmap_index = index % BITS_PER_LONG;

	clear_bit(bitmap_index, &uxu_rtn->is_file_dirty_bitmaps[list_index]);
}

static inline void
uvm_uxu_block_set_file_dirty(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	size_t index = uvm_va_range_block_index(va_range, va_block->start);
	size_t list_index = index / BITS_PER_LONG;
	size_t bitmap_index = index % BITS_PER_LONG;

	set_bit(bitmap_index, &uxu_rtn->is_file_dirty_bitmaps[list_index]);
}

static inline bool
uvm_uxu_need_to_evict_from_gpu(uvm_va_block_t *va_block)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_block->va_range->node.uxu_rtn;

	return (uxu_rtn->flags & UVM_UXU_FLAG_WRITE)
	    || (uxu_rtn->flags & UVM_UXU_FLAG_USEHOSTBUF);
}

/**
 * Mark that we just touch this block, which has in-buffer data.
 *
 * @param va_block: va_block to be marked.
 */
static inline void
uvm_uxu_block_mark_recent_in_buffer(uvm_va_block_t *va_block)
{
	uvm_uxu_va_space_t *uxu_va_space = &va_block->va_range->va_space->uxu_va_space;

	// Move this block to the tail of the LRU list.
	// mutex locking is commented out. It has been already held.
        uvm_mutex_lock(&uxu_va_space->lock_blocks);
	if (!list_empty(&va_block->uxu_lru))
		list_move_tail(&va_block->uxu_lru, &uxu_va_space->lru_head);
        uvm_mutex_unlock(&uxu_va_space->lock_blocks);
}

#endif
