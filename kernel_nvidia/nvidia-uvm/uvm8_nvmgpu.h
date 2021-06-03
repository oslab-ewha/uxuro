#ifndef __UVM8_NVMGPU_H__
#define __UVM8_NVMGPU_H__

#include "uvm8_va_space.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"

// Flags for each mapping
#define UVM_NVMGPU_FLAG_READ        0x01
#define UVM_NVMGPU_FLAG_WRITE       0x02
#define UVM_NVMGPU_FLAG_DONTTRASH   0x08
#define UVM_NVMGPU_FLAG_VOLATILE    0x10
#define UVM_NVMGPU_FLAG_USEHOSTBUF  0x20

NV_STATUS uvm_nvmgpu_initialize(uvm_va_space_t *va_space,
				unsigned long trash_nr_blocks,
				unsigned long trash_reserved_nr_pages,
				unsigned short flags);
NV_STATUS uvm_nvmgpu_register_file_va_space(uvm_va_space_t *va_space,
					    UVM_NVMGPU_REGISTER_FILE_VA_SPACE_PARAMS *params);
NV_STATUS uvm_nvmgpu_remap(uvm_va_space_t *va_space,
			   UVM_NVMGPU_REMAP_PARAMS *params);
NV_STATUS uvm_nvmgpu_unregister_va_range(uvm_va_range_t *va_range);

NV_STATUS uvm_nvmgpu_flush_host_block(uvm_va_space_t *va_space,
				      uvm_va_range_t *va_range,
				      uvm_va_block_t *va_block, bool is_evict,
				      const uvm_page_mask_t *page_mask);
NV_STATUS uvm_nvmgpu_flush_block(uvm_va_block_t *va_block);
NV_STATUS uvm_nvmgpu_flush(uvm_va_range_t *va_range);
NV_STATUS uvm_nvmgpu_release_block(uvm_va_block_t *va_block);

NV_STATUS uvm_nvmgpu_read_begin(uvm_va_block_t *va_block,
				uvm_va_block_retry_t *block_retry,
				uvm_service_block_context_t *service_context);
NV_STATUS uvm_nvmgpu_read_end(uvm_va_block_t *va_block);

NV_STATUS uvm_nvmgpu_write_begin(uvm_va_block_t *va_block, bool is_flush);
NV_STATUS uvm_nvmgpu_write_end(uvm_va_block_t *va_block, bool is_flush);

NV_STATUS uvm_nvmgpu_reduce_memory_consumption(uvm_va_space_t *va_space);

NV_STATUS uvm_nvmgpu_prepare_block_for_hostbuf(uvm_va_block_t *va_block);

void uvm_nvmgpu_set_page_dirty(struct page *page);

struct page *assign_pagecache(uvm_va_block_t * block, uvm_page_index_t page_index);

/**
 * Is this va_range managed by nvmgpu driver?
 *
 * @param va_range: va_range to be examined.
 * @return: true if this va_range is managed by nvmgpu driver, false otherwise.
 */
static inline bool
uvm_nvmgpu_is_managed(uvm_va_range_t *va_range)
{
	return va_range->node.nvmgpu_rtn.filp != NULL;
}

/**
 * Determine if we need to reclaim some blocks or not.
 *
 * @param nvmgpu_va_space: the va_space information related to NVMGPU.
 *
 * @return: true if we need to reclaim, false otherwise.
 */
static inline bool
uvm_nvmgpu_has_to_reclaim_blocks(uvm_nvmgpu_va_space_t *
						    nvmgpu_va_space)
{
	unsigned long	freeram = global_zone_page_state(NR_FREE_PAGES);
	unsigned long	pagecacheram = global_zone_page_state(NR_FILE_PAGES);
	return freeram + pagecacheram < nvmgpu_va_space->trash_reserved_nr_pages;
}

static inline bool
uvm_nvmgpu_block_file_dirty(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	return test_bit(bitmap_index, &nvmgpu_rtn->is_file_dirty_bitmaps[list_index]);
}

static inline bool
uvm_nvmgpu_need_to_copy_from_file(uvm_va_block_t *va_block,
						     uvm_processor_id_t
						     processor_id)
{
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_block->va_range->node.nvmgpu_rtn;

	if (!uvm_nvmgpu_is_managed(va_block->va_range))
		return false;

	if (uvm_nvmgpu_block_file_dirty(va_block))
		return true;

	return (!(nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_VOLATILE)	&&
		!((nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_USEHOSTBUF) && va_block->nvmgpu_use_uvm_buffer) &&
		((nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_READ) || UVM_ID_IS_CPU(processor_id)));
}

static inline void
uvm_nvmgpu_block_clear_has_data(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	clear_bit(bitmap_index, &nvmgpu_rtn->has_data_bitmaps[list_index]);
}

static inline void
uvm_nvmgpu_block_set_has_data(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	set_bit(bitmap_index, &nvmgpu_rtn->has_data_bitmaps[list_index]);
}

static inline bool
uvm_nvmgpu_block_has_data(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

	size_t	index = uvm_va_range_block_index(va_range, va_block->start);
	size_t	list_index = index / BITS_PER_LONG;
	size_t	bitmap_index = index % BITS_PER_LONG;

	return test_bit(bitmap_index, &nvmgpu_rtn->has_data_bitmaps[list_index]);
}

static inline void
uvm_nvmgpu_block_clear_file_dirty(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

	size_t index = uvm_va_range_block_index(va_range, va_block->start);
	size_t list_index = index / BITS_PER_LONG;
	size_t bitmap_index = index % BITS_PER_LONG;

	clear_bit(bitmap_index, &nvmgpu_rtn->is_file_dirty_bitmaps[list_index]);
}

static inline void
uvm_nvmgpu_block_set_file_dirty(uvm_va_block_t *va_block)
{
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

	size_t index = uvm_va_range_block_index(va_range, va_block->start);
	size_t list_index = index / BITS_PER_LONG;
	size_t bitmap_index = index % BITS_PER_LONG;

	set_bit(bitmap_index, &nvmgpu_rtn->is_file_dirty_bitmaps[list_index]);
}

static inline bool
uvm_nvmgpu_need_to_evict_from_gpu(uvm_va_block_t *va_block)
{
	uvm_nvmgpu_range_tree_node_t	*nvmgpu_rtn = &va_block->va_range->node.nvmgpu_rtn;

	return (nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_WRITE)
	    || (nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_USEHOSTBUF);
}

/**
 * Mark that we just touch this block, which has in-buffer data.
 *
 * @param va_block: va_block to be marked.
 */
static inline void
uvm_nvmgpu_block_mark_recent_in_buffer(uvm_va_block_t *va_block)
{
	uvm_nvmgpu_va_space_t *nvmgpu_va_space = &va_block->va_range->va_space->nvmgpu_va_space;

	// Move this block to the tail of the LRU list.
	// mutex locking is commented out. It has been already held.
//    uvm_mutex_lock(&nvmgpu_va_space->lock);
	list_move_tail(&va_block->nvmgpu_lru, &nvmgpu_va_space->lru_head);
//    uvm_mutex_unlock(&nvmgpu_va_space->lock);
}

#endif
