#include <linux/syscalls.h>
#include <linux/delay.h>
#include <linux/aio.h>
#include <linux/swap.h>
#include <linux/writeback.h>
#include <linux/fs.h>
#include <linux/backing-dev.h>
#include <linux/uio.h>

#include "nv_uvm_interface.h"
#include "uvm8_api.h"
#include "uvm8_channel.h"
#include "uvm8_global.h"
#include "uvm8_gpu.h"
#include "uvm8_gpu_semaphore.h"
#include "uvm8_hal.h"
#include "uvm8_procfs.h"
#include "uvm8_pmm_gpu.h"
#include "uvm8_va_space.h"
#include "uvm8_gpu_replayable_faults.h"
#include "uvm8_user_channel.h"
#include "uvm8_perf_events.h"
#include "uvm_common.h"
#include "ctrl2080mc.h"
#include "nv-kthread-q.h"
#include "uvm_linux.h"
#include "uvm_common.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"
#include "uvm8_hal_types.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_push.h"
#include "uvm8_perf_thrashing.h"
#include "uvm8_perf_prefetch.h"
#include "uvm8_mem.h"
#include "uvm8_uxu.h"

static void uvm_page_mask_fill(uvm_page_mask_t *mask)
{
	bitmap_fill(mask->bitmap, PAGES_PER_UVM_VA_BLOCK);
}

#define MIN(x,y) (x < y ? x : y)

static void *fsdata_array[PAGES_PER_UVM_VA_BLOCK];

static int pagecache_reducer(void *ctx)
{
	uvm_va_space_t *va_space = (uvm_va_space_t *)ctx;
	uvm_uxu_va_space_t *uxu_va_space = &va_space->uxu_va_space;
	uvm_thread_context_wrapper_t	thread_context;

	uvm_thread_context_add(&thread_context.context);

	while (!kthread_should_stop()) {
		if (uvm_uxu_has_to_reclaim_blocks(uxu_va_space))
			uvm_uxu_reduce_memory_consumption(va_space);
		schedule_timeout_idle(10);
	}

	uvm_thread_context_remove(&thread_context.context);
	return 0;
}

void
stop_pagecache_reducer(uvm_va_space_t *va_space)
{
	uvm_uxu_va_space_t *uxu_va_space = &va_space->uxu_va_space;
	if (uxu_va_space->reducer) {
		kthread_stop(uxu_va_space->reducer);
		uxu_va_space->reducer = NULL;
	}
}

/**
 * Initialize the UXU module. This function has to be called once per
 * va_space. It must be called before calling
 * "uvm_uxu_register_file_va_space"
 *
 * @param va_space: va_space to be initialized this module with.
 *
 * @param trash_nr_blocks: maximum number of va_block UXU should evict out
 * at one time.
 *
 * @param trash_reserved_nr_pages: UXU will automatically evicts va_block
 * when number of free pages plus number of page-cache pages less than this
 * value.
 *
 * @param flags: the flags that dictate the optimization behaviors. See
 * UVM_UXU_INIT_* for more details.
 *
 * @return: NV_ERR_INVALID_OPERATION if `va_space` has been initialized already,
 * otherwise NV_OK.
 */
NV_STATUS
uvm_uxu_initialize(uvm_va_space_t *va_space, unsigned long trash_nr_blocks, unsigned long trash_reserved_nr_pages, unsigned short flags)
{
	uvm_uxu_va_space_t *uxu_va_space = &va_space->uxu_va_space;

	if (!uxu_va_space->is_initailized) {
		INIT_LIST_HEAD(&uxu_va_space->lru_head);
		/* TODO: Lower down the locking order.
		 * Because invalid locking order warnings are generated when debug mode is enabled.
		 */
		uvm_mutex_init(&uxu_va_space->lock, UVM_LOCK_ORDER_VA_SPACE);
		uvm_mutex_init(&uxu_va_space->lock_blocks, UVM_LOCK_ORDER_VA_SPACE_UXU);
		uxu_va_space->trash_nr_blocks = trash_nr_blocks;
		uxu_va_space->trash_reserved_nr_pages = trash_reserved_nr_pages;
		uxu_va_space->flags = flags;
		uxu_va_space->is_initailized = true;

		uxu_va_space->reducer = kthread_run(pagecache_reducer, va_space, "reducer");
		return NV_OK;
	}
	else
		return NV_ERR_INVALID_OPERATION;
}

/**
 * Register a file to this `va_space`.
 * UXU will start tracking this UVM region if this function return success.
 *
 * @param va_space: va_space to register the file to.
 *
 * @param params: register parameters containing info about the file, size, etc.
 *
 * @return: NV_OK on success, NV_ERR_* otherwise.
 */
NV_STATUS
uvm_uxu_map(uvm_va_space_t *va_space, UVM_UXU_MAP_PARAMS *params)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn;

	uvm_range_tree_node_t	*node = uvm_range_tree_find(&va_space->va_range_tree, (NvU64)params->uvm_addr);
	NvU64	expected_start_addr = (NvU64)params->uvm_addr;
	NvU64	expected_end_addr = expected_start_addr + params->size - 1;

	size_t	max_nr_blocks;

	// Make sure that uvm_uxu_initialize is called before this function.
	if (!va_space->uxu_va_space.is_initailized) {
		printk(KERN_DEBUG "Error: Call uvm_uxu_register_file_va_space before uvm_uxu_initialize\n");
		return NV_ERR_INVALID_OPERATION;
	}

	// Find uvm node associated with the specified UVM address range.
	if (node == NULL) {
		printk(KERN_DEBUG "Error: no matching va range for 0x%llx-0x%llx\n", expected_start_addr, expected_end_addr);
		return NV_ERR_OPERATING_SYSTEM;
	}
	// It would be OK if a va range includes the UVM address range.
	if (node->end < expected_end_addr) {
		printk(KERN_DEBUG "Cannot find uvm range 0x%llx - 0x%llx\n", expected_start_addr, expected_end_addr);
		if (node)
			printk(KERN_DEBUG "Closet uvm range 0x%llx - 0x%llx\n", node->start, node->end);
		return NV_ERR_OPERATING_SYSTEM;
	}

	uxu_rtn = &node->uxu_rtn;

	// Get the struct file from the input file descriptor.
	if ((uxu_rtn->filp = fget(params->backing_fd)) == NULL) {
		printk(KERN_DEBUG "Cannot find the backing fd: %d\n", params->backing_fd);
		return NV_ERR_OPERATING_SYSTEM;
	}

	// Record the flags and the file size.
	uxu_rtn->flags = params->flags;
	uxu_rtn->size = params->size;

	// Calculate the number of blocks associated with this UVM range.
	max_nr_blocks = uvm_va_range_num_blocks(container_of(node, uvm_va_range_t, node));

	uxu_rtn->iov = kmalloc(sizeof(struct iovec) * PAGES_PER_UVM_VA_BLOCK, GFP_KERNEL);
	if (!uxu_rtn->iov) {
		fput(uxu_rtn->filp);
		uxu_rtn->filp = NULL;
		return NV_ERR_NO_MEMORY;
	}

	return NV_OK;
}

NV_STATUS
uvm_uxu_remap(uvm_va_space_t *va_space, UVM_UXU_REMAP_PARAMS *params)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn;
	uvm_va_block_t	*va_block, *va_block_next;
	uvm_uxu_va_space_t	*uxu_va_space = &va_space->uxu_va_space;

	uvm_va_range_t	*va_range = uvm_va_range_find(va_space, (NvU64)params->uvm_addr);
	NvU64	expected_start_addr = (NvU64)params->uvm_addr;

	// Make sure that uvm_uxu_initialize is called before this function.
	if (!va_space->uxu_va_space.is_initailized) {
		printk(KERN_DEBUG "Error: Call uvm_uxu_remap before uvm_uxu_initialize\n");
		return NV_ERR_INVALID_OPERATION;
	}

	if (!va_range || va_range->node.start != expected_start_addr) {
		printk(KERN_DEBUG "Cannot find uvm whose address starts from 0x%llx\n", expected_start_addr);
		if (va_range)
			printk(KERN_DEBUG "Closet uvm range 0x%llx - 0x%llx\n", va_range->node.start, va_range->node.end);
		return NV_ERR_OPERATING_SYSTEM;
	}

	uxu_rtn = &va_range->node.uxu_rtn;

	if (uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE)
		uvm_mutex_lock(&uxu_va_space->lock);

	// Volatile data is simply discarded even though it has been remapped with non-volatile
	for_each_va_block_in_va_range_safe(va_range, va_block, va_block_next) {
		va_block->is_dirty = false;
		if (uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE) {
			uvm_uxu_release_block(va_block);
			list_del(&va_block->uxu_lru);
		}
	}

	if (uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE)
		uvm_mutex_unlock(&uxu_va_space->lock);

	uxu_rtn->flags = params->flags;

	return NV_OK;
}

/**
 * Unregister the specified va_range.
 * UXU will stop tracking this `va_range` after this point.
 *
 * @param va_range: va_range to be untracked.
 *
 * @return: always NV_OK.
 */
NV_STATUS
uvm_uxu_unregister_va_range(uvm_va_range_t *va_range)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;
	struct file	*filp = uxu_rtn->filp;

	UVM_ASSERT(filp != NULL);

	if ((va_range->node.uxu_rtn.flags & UVM_UXU_FLAG_WRITE) && !(va_range->node.uxu_rtn.flags & UVM_UXU_FLAG_VOLATILE)) {
		uvm_uxu_flush(va_range);
        }

	if (uxu_rtn->iov)
		kfree(uxu_rtn->iov);

	if ((uxu_rtn->flags & UVM_UXU_FLAG_WRITE) && !(uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE))
		vfs_fsync(filp, 1);

	fput(filp);

	return NV_OK;
}

static void
uvm_uxu_unmap_page(uvm_va_block_t *va_block, int page_index)
{
	uvm_gpu_id_t	id;

	for_each_gpu_id(id) {
		uvm_gpu_t *gpu;
		uvm_va_block_gpu_state_t *gpu_state = va_block->gpus[uvm_id_gpu_index(id)];
		if (!gpu_state)
			continue;

		if (gpu_state->cpu_pages_dma_addrs[page_index] == 0)
			continue;

		UVM_ASSERT(va_block->va_range);
		UVM_ASSERT(va_block->va_range->va_space);
		gpu = uvm_va_space_get_gpu(va_block->va_range->va_space, id);

		uvm_gpu_unmap_cpu_page(gpu, gpu_state->cpu_pages_dma_addrs[page_index]);
		gpu_state->cpu_pages_dma_addrs[page_index] = 0;
	}
}

static NV_STATUS
insert_pagecache_to_va_block(uvm_va_block_t *va_block, int page_id, struct page *page)
{
	NV_STATUS	status = NV_OK;
	uvm_gpu_id_t	gpu_id;

	lock_page(page);

	if (va_block->cpu.pages[page_id] != page) {
		if (va_block->cpu.pages[page_id] != NULL) {
			uvm_uxu_unmap_page(va_block, page_id);
			if (uvm_page_mask_test(&va_block->cpu.pagecached, page_id))
				put_page(va_block->cpu.pages[page_id]);
			else
				__free_page(va_block->cpu.pages[page_id]);
		}
		for_each_gpu_id(gpu_id) {
			uvm_gpu_t	*gpu;
			uvm_va_block_gpu_state_t	*gpu_state = va_block->gpus[uvm_id_gpu_index(gpu_id)];
			if (!gpu_state)
				continue;

			UVM_ASSERT(gpu_state->cpu_pages_dma_addrs[page_id] == 0);

			UVM_ASSERT(va_block->va_range);
			UVM_ASSERT(va_block->va_range->va_space);
			gpu = uvm_va_space_get_gpu(va_block->va_range->va_space, gpu_id);

			status = uvm_gpu_map_cpu_pages(gpu, page, PAGE_SIZE, &gpu_state->cpu_pages_dma_addrs[page_id]);
			if (status != NV_OK) {
				printk(KERN_DEBUG "Cannot do uvm_gpu_map_cpu_pages\n");
				goto insert_pagecache_to_va_block_error;
			}
		}
		va_block->cpu.pages[page_id] = page;
	}
	else {
		put_page(page);
	}

	uvm_page_mask_set(&va_block->cpu.pagecached, page_id);

	return NV_OK;

insert_pagecache_to_va_block_error:
	uvm_uxu_unmap_page(va_block, page_id);
	unlock_page(page);

	return status;
}

/**
 * Inspired by generic_file_buffered_read in /mm/filemap.c.
 */
static int
prepare_page_for_read(struct file *filp, loff_t ppos, uvm_va_block_t *va_block, int page_id)
{
	struct address_space	*mapping = filp->f_mapping;
	struct inode	*inode = mapping->host;
	struct file_ra_state	*ra = &filp->f_ra;
	pgoff_t	index;
	pgoff_t	last_index;
	pgoff_t	prev_index;
	unsigned long	offset;      /* offset into pagecache page */
	unsigned int	prev_offset;
	int	error = 0;

	index = ppos >> PAGE_SHIFT;
	prev_index = ra->prev_pos >> PAGE_SHIFT;
	prev_offset = ra->prev_pos & (PAGE_SIZE-1);
	last_index = (ppos + PAGE_SIZE + PAGE_SIZE-1) >> PAGE_SHIFT;
	offset = ppos & ~PAGE_MASK;

	for (;;) {
		struct page	*page;
		pgoff_t	end_index;
		loff_t	isize;
		unsigned long	nr;
		NV_STATUS	ret;

		cond_resched();
find_page:
		if (fatal_signal_pending(current)) {
			error = -EINTR;
			goto out;
		}

		page = find_get_page(mapping, index);
		if (!page) {
			page_cache_sync_readahead(mapping, ra, filp, index, last_index - index);
			page = find_get_page(mapping, index);
			if (unlikely(page == NULL))
				goto no_cached_page;
		}
		if (PageReadahead(page)) {
			page_cache_async_readahead(mapping, ra, filp, page, index, last_index - index);
		}
		if (!PageUptodate(page)) {
			/*
			 * See comment in do_read_cache_page on why
			 * wait_on_page_locked is used to avoid unnecessarily
			 * serialisations and why it's safe.
			 */
			error = wait_on_page_locked_killable(page);
			if (unlikely(error))
				goto readpage_error;
			if (PageUptodate(page))
				goto page_ok;

			if (inode->i_blkbits == PAGE_SHIFT ||
			    !mapping->a_ops->is_partially_uptodate)
				goto page_not_up_to_date;
			if (!trylock_page(page))
				goto page_not_up_to_date;
			/* Did it get truncated before we got the lock? */
			if (!page->mapping)
				goto page_not_up_to_date_locked;
			if (!mapping->a_ops->is_partially_uptodate(page, offset, PAGE_SIZE))
				goto page_not_up_to_date_locked;
			unlock_page(page);
		}
page_ok:
		/*
		 * i_size must be checked after we know the page is Uptodate.
		 *
		 * Checking i_size after the check allows us to calculate
		 * the correct value for "nr", which means the zero-filled
		 * part of the page is not copied back to userspace (unless
		 * another truncate extends the file - this is desired though).
		 */

		isize = i_size_read(inode);
		end_index = (isize - 1) >> PAGE_SHIFT;
		if (unlikely(!isize || index > end_index)) {
			put_page(page);
			goto out;
		}

		/* nr is the maximum number of bytes to copy from this page */
		nr = PAGE_SIZE;
		if (index == end_index) {
			nr = ((isize - 1) & ~PAGE_MASK) + 1;
			if (nr <= offset) {
				put_page(page);
				goto out;
			}
		}
		nr = nr - offset;

		/* If users can be writing to this page using arbitrary
		 * virtual addresses, take care about potential aliasing
		 * before reading the page on the kernel side.
		 */
		if (mapping_writably_mapped(mapping))
			flush_dcache_page(page);

		/*
		 * When a sequential read accesses a page several times,
		 * only mark it as accessed the first time.
		 */
		if (prev_index != index || offset != prev_offset)
			mark_page_accessed(page);
		prev_index = index;

		/*
		 * Ok, we have the page, and it's up-to-date, so
		 * now we can insert it to the va_block...
		 */
		ret = insert_pagecache_to_va_block(va_block, page_id, page);
		if (ret != NV_OK) {
			error = ret;
			goto out;
		}

		offset += PAGE_SIZE;
		index += offset >> PAGE_SHIFT;
		offset &= ~PAGE_MASK;
		prev_offset = offset;

		goto out;

page_not_up_to_date:
		/* Get exclusive access to the page ... */
		error = lock_page_killable(page);
		if (unlikely(error))
			goto readpage_error;

page_not_up_to_date_locked:
		/* Did it get truncated before we got the lock? */
		if (!page->mapping) {
			unlock_page(page);
			put_page(page);
			continue;
		}

		/* Did somebody else fill it already? */
		if (PageUptodate(page)) {
			unlock_page(page);
			goto page_ok;
		}

readpage:
		/*
		 * A previous I/O error may have been due to temporary
		 * failures, eg. multipath errors.
		 * PG_error will be set again if readpage fails.
		 */
		ClearPageError(page);
		/* Start the actual read. The read will unlock the page. */
		error = mapping->a_ops->readpage(filp, page);

		if (unlikely(error)) {
			if (error == AOP_TRUNCATED_PAGE) {
				put_page(page);
				error = 0;
				goto find_page;
			}
			goto readpage_error;
		}

		if (!PageUptodate(page)) {
			error = lock_page_killable(page);
			if (unlikely(error))
				goto readpage_error;
			if (!PageUptodate(page)) {
				if (page->mapping == NULL) {
					/*
					 * invalidate_mapping_pages got it
					 */
					unlock_page(page);
					put_page(page);
					goto find_page;
				}
				unlock_page(page);
				ra->ra_pages /= 4;
				error = -EIO;
				goto readpage_error;
			}
			unlock_page(page);
		}

		goto page_ok;

readpage_error:
		/* UHHUH! A synchronous read error occurred. Report it */
		put_page(page);
		goto out;

no_cached_page:
		/*
		 * Ok, it wasn't cached, so we need to create a new
		 * page..
		 */
		page = page_cache_alloc(mapping);
		if (!page) {
			error = -ENOMEM;
			goto out;
		}
		error = add_to_page_cache_lru(page, mapping, index,
					      mapping_gfp_constraint(mapping, GFP_KERNEL));
		if (error) {
			put_page(page);
			if (error == -EEXIST) {
				error = 0;
				goto find_page;
			}
			goto out;
		}
		goto readpage;
	}

	error = -EAGAIN;
out:
	ra->prev_pos = prev_index;
	ra->prev_pos <<= PAGE_SHIFT;
	ra->prev_pos |= prev_offset;

	file_accessed(filp);
	return error;
}

static struct page *
assign_page(uvm_va_block_t *block, bool zero)
{
	struct page *page;
	gfp_t gfp_flags;

	gfp_flags = NV_UVM_GFP_FLAGS | GFP_HIGHUSER;
	if (zero)
		gfp_flags |= __GFP_ZERO;

	page = alloc_pages(gfp_flags, 0);
	if (!page) {
		return NULL;
	}

	// the kernel has 'written' zeros to this page, so it is dirty
	if (zero)
		SetPageDirty(page);

	return page;
}

static struct page *
assign_pagecache(uvm_va_block_t *block, uvm_page_index_t page_index)
{
	uvm_va_range_t	*va_range = block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;
	struct file	*uxu_file = uxu_rtn->filp;
	loff_t	file_start_offset = block->start - block->va_range->node.start;
	loff_t	offset;

	offset = file_start_offset + page_index * PAGE_SIZE;
	return read_mapping_page(uxu_file->f_mapping, offset, NULL);
}

static inline bool
uxu_is_pagecachable(uvm_va_block_t *block, uvm_page_index_t page_id)
{
	struct inode *inode;
	uvm_page_index_t	outer_max;
	loff_t len_remain;

	if (block->va_range->node.uxu_rtn.flags & UVM_UXU_FLAG_VOLATILE)
		return false;
	inode = block->va_range->node.uxu_rtn.filp->f_mapping->host;
	len_remain = i_size_read(inode) - (block->start - block->va_range->node.start);
	outer_max = (len_remain + PAGE_SIZE - 1) >> PAGE_SHIFT;
	if (page_id >= outer_max)
		return false;
	return true;
}

struct page *
uxu_get_page(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero)
{
	struct page	*page;

	if (uxu_is_pagecachable(block, page_index)) {
		page = assign_pagecache(block, page_index);
		if (page)
			uvm_page_mask_set(&block->cpu.pagecached, page_index);
		else {
			page = assign_page(block, zero);
			if (page) {
				uvm_page_mask_clear(&block->cpu.pagecached, page_index);
			}
		}
	}
	else {
		page = assign_page(block, zero);
		if (page)
			uvm_page_mask_clear(&block->cpu.pagecached, page_index);
	}

	return page;
}

static bool
fill_pagecaches_for_read(struct file *uxu_file, uvm_va_block_t *va_block, uvm_va_block_region_t region)
{
	struct inode	*inode = uxu_file->f_mapping->host;
	loff_t	isize;
	uvm_page_mask_t read_mask;
	int page_id;
	// Calculate the file offset based on the block start address.
	loff_t	file_start_offset = va_block->start - va_block->va_range->node.start;

	uvm_page_mask_fill(&read_mask);

	isize = i_size_read(inode);

	// Fill in page-cache pages to va_block
	for_each_va_block_page_in_region_mask(page_id, &read_mask, region) {
		loff_t	offset = file_start_offset + page_id * PAGE_SIZE;

		if (unlikely(offset >= isize)) {
			struct page	*page = va_block->cpu.pages[page_id];
			if (page)
				lock_page(page);
			continue;
		}
		if (prepare_page_for_read(uxu_file, offset, va_block, page_id) != 0) {
			printk(KERN_DEBUG "Cannot prepare page for read at file offset 0x%llx\n", offset);
			return false;
		}
		UVM_ASSERT(va_block->cpu.pages[page_id]);
	}

	return true;
}

static uvm_page_index_t
get_region_readable_outer(uvm_va_block_t *va_block, struct file *uxu_file)
{
	uvm_page_index_t	outer = ((va_block->end - va_block->start) >> PAGE_SHIFT) + 1;
	uvm_page_index_t	outer_max;
	struct inode	*inode = uxu_file->f_mapping->host;
	loff_t	len_remain = i_size_read(inode) - (va_block->start - va_block->va_range->node.start);

	outer_max = (len_remain + PAGE_SIZE - 1) >> PAGE_SHIFT;
	if (outer > outer_max)
		return outer_max;
	return outer;
}

/**
 * Prepare page-cache pages to be read.
 *
 * @param va_block: data will be put in this va_block.
 *
 * @param block_retry: need this to allocate memory pages and register them to
 * this UVM range.
 *
 * @param service_context: need it the same as block_retry.
 *
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
NV_STATUS
uvm_uxu_read_begin(uvm_va_block_t *va_block, uvm_va_block_retry_t *block_retry, uvm_service_block_context_t *service_context)
{
	NV_STATUS	status = NV_OK;

	uvm_va_range_t	*va_range = va_block->va_range;

	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	struct file	*uxu_file = uxu_rtn->filp;

	// Specify that the entire block is the region of concern.
	uvm_va_block_region_t region = uvm_va_block_region(0, get_region_readable_outer(va_block, uxu_file));

	uvm_page_mask_t	my_mask;
	// Record the original page mask and set the mask to all 1s.
	uvm_page_mask_t	original_page_mask;

	uvm_page_mask_copy(&original_page_mask, &service_context->block_context.make_resident.page_mask);

	uvm_page_mask_init_from_region(&service_context->block_context.make_resident.page_mask, region, NULL);
	uvm_page_mask_copy(&my_mask, &service_context->block_context.make_resident.page_mask);

	UVM_ASSERT(uxu_file != NULL);

	// Change this va_block's state: all pages are the residents of CPU.
	status = uvm_va_block_make_resident(va_block,
					    block_retry,
					    &service_context->block_context,
					    UVM_ID_CPU,
					    region,
					    &my_mask,
					    NULL,
					    UVM_MAKE_RESIDENT_CAUSE_UXU);

	if (status != NV_OK) {
		printk(KERN_DEBUG "Cannot make temporary resident on CPU\n");
		goto read_begin_err_0;
	}

	status = uvm_tracker_wait(&va_block->tracker);
	if (status != NV_OK) {
		printk(KERN_DEBUG "Cannot make temporary resident on CPU\n");
		goto read_begin_err_0;
	}

	if (fill_pagecaches_for_read(uxu_file, va_block, region))
		va_block->has_data = true;
	else
		status = NV_ERR_OPERATING_SYSTEM;

read_begin_err_0:
	// Put back the original mask.
	uvm_page_mask_copy(&service_context->block_context.make_resident.page_mask, &original_page_mask);

	return status;
}

NV_STATUS
uvm_uxu_read_end(uvm_va_block_t *va_block)
{
	int	page_id;
	struct page	*page;

	uvm_page_mask_t	read_mask;

	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_block->va_range->node.uxu_rtn;
	struct file	*uxu_file = uxu_rtn->filp;
	uvm_va_block_region_t	region = uvm_va_block_region(0, get_region_readable_outer(va_block, uxu_file));

	uvm_page_mask_fill(&read_mask);
	for_each_va_block_page_in_region_mask(page_id, &read_mask, region) {
		page = va_block->cpu.pages[page_id];
		if (page)
			unlock_page(page);
	}

	return NV_OK;
}

void
uxu_try_load_block(uvm_va_block_t *block, uvm_va_block_retry_t *block_retry, uvm_service_block_context_t *service_context, uvm_processor_id_t processor_id)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn = &block->va_range->node.uxu_rtn;
	NV_STATUS	status;

	if (block->has_data)
		return;
	if (uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE)
		return;
	if (!(uxu_rtn->flags & UVM_UXU_FLAG_READ) && !UVM_ID_IS_CPU(processor_id))
		return;

	status = uvm_uxu_read_begin(block, block_retry, service_context);
	if (status != NV_OK)
		return;

	uvm_tracker_wait(&block->tracker);
	uvm_uxu_read_end(block);

        if (UVM_ID_IS_CPU(processor_id)) {
		uvm_uxu_write_begin(block, false);
		uvm_uxu_write_end(block, false);
        }
	uvm_uxu_block_mark_recent_in_buffer(block);
}

/**
 * Evict out the block. This function can handle both CPU-only and GPU blocks.
 *
 * @param va_block: the block to be evicted.
 *
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
NV_STATUS
uvm_uxu_flush_block(uvm_va_block_t *va_block)
{
	NV_STATUS	status = NV_OK;
	uvm_va_range_t	*va_range = va_block->va_range;
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_range->node.uxu_rtn;

	if (!(uxu_rtn->flags & UVM_UXU_FLAG_WRITE))
		return NV_OK;

	// Move data from GPU to CPU
	if (uvm_processor_mask_get_gpu_count(&(va_block->resident)) > 0) {
		uvm_va_block_region_t region = uvm_va_block_region_from_block(va_block);
		uvm_va_block_context_t *block_context = uvm_va_block_context_alloc();

		if (!block_context) {
			printk(KERN_DEBUG "NV_ERR_NO_MEMORY\n");
			return NV_ERR_NO_MEMORY;
		}

		uvm_mutex_lock(&va_block->lock);
		// Move data resided on the GPU to host.
		status = uvm_va_block_migrate_locked(va_block, NULL, block_context, region, UVM_ID_CPU, UVM_MIGRATE_MODE_MAKE_RESIDENT, NULL);
		uvm_mutex_unlock(&va_block->lock);

		uvm_va_block_context_free(block_context);

		if (status != NV_OK) {
			printk(KERN_DEBUG "NOT NV_OK\n");
			return status;
		}

		// Wait for the d2h transfer to complete.
		status = uvm_tracker_wait(&va_block->tracker);

		if (status != NV_OK) {
			printk(KERN_DEBUG "NOT NV_OK\n");
			return status;
		}
	}

	return status;
}

/**
 * Flush all blocks in the `va_range`.
 *
 * @param va_range: va_range that we want to flush the data.
 *
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
NV_STATUS
uvm_uxu_flush(uvm_va_range_t *va_range)
{
	NV_STATUS	status = NV_OK;
	uvm_va_block_t	*va_block, *va_block_next;

	// Evict blocks one by one.
	for_each_va_block_in_va_range_safe(va_range, va_block, va_block_next) {
		if ((status = uvm_uxu_flush_block(va_block)) != NV_OK) {
			printk(KERN_DEBUG "Encountered a problem with uvm_uxu_flush_block\n");
			break;
		}
	}

	return status;
}

/**
 * Free memory associated with the `va_block`.
 *
 * @param va_block: va_block to be freed.
 *
 * @return: always NV_OK;
 */
NV_STATUS
uvm_uxu_release_block(uvm_va_block_t *va_block)
{
	uvm_va_block_t	*old;
	size_t	index;

	uvm_va_range_t	*va_range = va_block->va_range;

	UVM_ASSERT(va_block != NULL);

	if (va_range == NULL) {
		/* maybe already destroyed ?? */
		return NV_OK;
	}

	// Remove the block from the list.
	index = uvm_va_range_block_index(va_range, va_block->start);
	old = (uvm_va_block_t *)nv_atomic_long_cmpxchg(&va_range->blocks[index], (long)va_block, (long)NULL);

	// Free the block.
	if (old == va_block) {
		uvm_va_block_kill(va_block);
	}

	return NV_OK;
}

NV_STATUS
uvm_uxu_write_begin(uvm_va_block_t *va_block, bool is_flush)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_block->va_range->node.uxu_rtn;
	int		page_id;
	NV_STATUS	status = NV_OK;

	// Calculate the file offset based on the block start address.
	loff_t	file_start_offset = va_block->start - va_block->va_range->node.start;
	loff_t	file_position;

	struct file	*uxu_file = uxu_rtn->filp;
	struct inode	*f_inode = file_inode(uxu_file);
	struct address_space	*mapping = uxu_file->f_mapping;
	struct inode	*m_inode = mapping->host;
	const struct address_space_operations	*a_ops = mapping->a_ops;

	struct page	*page;
	void	*fsdata;

	uvm_va_space_t	*va_space;

	UVM_ASSERT(va_block->va_range);
	UVM_ASSERT(va_block->va_range->va_space);
	va_space = va_block->va_range->va_space;

	inode_lock(f_inode);

	current->backing_dev_info = inode_to_bdi(m_inode);

	file_remove_privs(uxu_file);

	file_update_time(uxu_file);

	for_each_va_block_page(page_id, va_block) {
		uvm_gpu_id_t id;
		long f_status = 0;

		file_position = file_start_offset + page_id * PAGE_SIZE;

		if (file_position >= uxu_rtn->size)
			break;

		f_status = a_ops->write_begin(uxu_file, mapping, file_position,
					      MIN(PAGE_SIZE, uxu_rtn->size - file_position), 0, &page, &fsdata);

		if (f_status != 0 || page == NULL)
			continue;

		if (mapping_writably_mapped(mapping))
			flush_dcache_page(page);

		fsdata_array[page_id] = fsdata;

		if (va_block->cpu.pages[page_id] != NULL)
			uvm_uxu_unmap_page(va_block, page_id);

		for_each_gpu_id(id) {
			uvm_gpu_t *gpu;
			uvm_va_block_gpu_state_t *gpu_state = va_block->gpus[uvm_id_gpu_index(id)];
			if (!gpu_state)
				continue;

			UVM_ASSERT(gpu_state->cpu_pages_dma_addrs[page_id] == 0);

			gpu = uvm_va_space_get_gpu(va_space, id);

			status = uvm_gpu_map_cpu_pages(gpu, page, PAGE_SIZE, &gpu_state->cpu_pages_dma_addrs[page_id]);
			UVM_ASSERT(status == NV_OK);

			uvm_pmm_sysmem_mappings_remove_gpu_mapping_on_eviction(&gpu->pmm_sysmem_mappings, gpu_state->cpu_pages_dma_addrs[page_id]);
			status = uvm_pmm_sysmem_mappings_add_gpu_mapping(&gpu->pmm_sysmem_mappings,
									 gpu_state->cpu_pages_dma_addrs[page_id],
									 uvm_va_block_cpu_page_address(va_block, page_id),
									 PAGE_SIZE,
									 va_block,
									 UVM_ID_CPU);
			UVM_ASSERT(status == NV_OK);
		}

		if (va_block->cpu.pages[page_id] != page) {
			if (va_block->cpu.pages[page_id]) {
				if (uvm_page_mask_test(&va_block->cpu.pagecached, page_id))
					put_page(va_block->cpu.pages[page_id]);
				else
					__free_page(va_block->cpu.pages[page_id]);
			}
			va_block->cpu.pages[page_id] = page;
			get_page(page);
		}
		uvm_page_mask_set(&va_block->cpu.pagecached, page_id);
	}

	return status;
}

NV_STATUS
uvm_uxu_write_end(uvm_va_block_t *va_block, bool is_flush)
{
	NV_STATUS	status = NV_OK;

	uvm_uxu_range_tree_node_t	*uxu_rtn = &va_block->va_range->node.uxu_rtn;
	struct file	*uxu_file = uxu_rtn->filp;
	struct inode	*f_inode = file_inode(uxu_file);
	struct address_space	*mapping = uxu_file->f_mapping;
	const struct address_space_operations	*a_ops = mapping->a_ops;

	int	page_id;

	loff_t	file_start_offset = va_block->start - va_block->va_range->node.start;
	loff_t	file_position;

	for_each_va_block_page(page_id, va_block) {
		struct page *page = va_block->cpu.pages[page_id];
		void *fsdata = fsdata_array[page_id];

		file_position = file_start_offset + page_id * PAGE_SIZE;

		if (file_position >= uxu_rtn->size)
			break;

		if (page) {
			size_t bytes = MIN(PAGE_SIZE, uxu_rtn->size - file_position);
			flush_dcache_page(page);
			mark_page_accessed(page);

			a_ops->write_end(uxu_file, mapping, file_position, bytes,
					 bytes, page, fsdata);

			balance_dirty_pages_ratelimited(mapping);
		}
	}

	va_block->is_dirty = true;

	current->backing_dev_info = NULL;

	inode_unlock(f_inode);

	return status;
}

NV_STATUS
uxu_va_block_make_resident(uvm_va_block_t *va_block,
			   uvm_va_block_retry_t *va_block_retry,
			   uvm_va_block_context_t *va_block_context,
			   uvm_processor_id_t dest_id,
			   uvm_va_block_region_t region,
			   const uvm_page_mask_t *page_mask,
			   const uvm_page_mask_t *prefetch_page_mask,
			   uvm_make_resident_cause_t cause)
{
	bool	do_uxu_write = false;
	NV_STATUS	status;

	if (!uvm_uxu_is_managed(va_block->va_range))
		return uvm_va_block_make_resident(va_block, va_block_retry, va_block_context, dest_id, region, page_mask, prefetch_page_mask, cause);

	if (uvm_uxu_need_to_evict_from_gpu(va_block) &&
	    (cause == UVM_MAKE_RESIDENT_CAUSE_EVICTION || (cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE && UVM_ID_IS_CPU(dest_id)))) {
		uvm_uxu_range_tree_node_t *uxu_rtn = &va_block->va_range->node.uxu_rtn;

		if (!va_block->is_dirty && (uxu_rtn->flags & UVM_UXU_FLAG_VOLATILE)) {
			uvm_uxu_block_mark_recent_in_buffer(va_block);
		}
		else {
			uvm_uxu_write_begin(va_block, cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);
			do_uxu_write = true;
		}
	}
	status = uvm_va_block_make_resident(va_block, va_block_retry, va_block_context, dest_id, region, page_mask, prefetch_page_mask, cause);

	status = uvm_tracker_wait(&va_block->tracker);
	uvm_uxu_write_end(va_block, cause == UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);

	return status;
}

/**
 * Automatically reduce memory usage if we need to.
 *
 * @param va_space: va_space that governs this operation.
 * @param force: if true, we will evict some blocks without checking for the memory pressure.
 *
 * @return: NV_OK on success, NV_ERR_* otherwise.
 */
NV_STATUS
uvm_uxu_reduce_memory_consumption(uvm_va_space_t *va_space)
{
	/*
	 * TODO: locking assertion failed. write lock is required.
	 */
	NV_STATUS	status = NV_OK;

	uvm_uxu_va_space_t	*uxu_va_space = &va_space->uxu_va_space;

	unsigned long	counter = 0;

	uvm_va_block_t	*va_block;
	struct list_head *lp, *next;

	uvm_va_space_down_write(va_space);
	// Reclaim blocks based on least recent transfer.

	list_for_each_safe(lp, next, &uxu_va_space->lru_head) {
		if (counter >= uxu_va_space->trash_nr_blocks)
			break;
		va_block = list_entry(lp, uvm_va_block_t, uxu_lru);

		// Terminate the loop since we cannot trash out blocks that have a copy on GPU
		if (uvm_processor_mask_get_gpu_count(&(va_block->resident)) > 0) {
			//printk(KERN_DEBUG "Encounter a block whose data are in GPU!!!\n");
			continue;
		}
		// Evict the block if it is on CPU only and this `va_range` has the write flag.
		if (uvm_processor_mask_get_count(&(va_block->resident)) > 0 && va_block->va_range->node.uxu_rtn.flags & UVM_UXU_FLAG_WRITE) {
			status = uvm_uxu_flush_host_block(va_block->va_range->va_space, va_block->va_range, va_block, true, NULL);
			if (status != NV_OK) {
				printk(KERN_DEBUG "Cannot evict block\n");
				continue;
			}
		}

		uvm_mutex_lock(&uxu_va_space->lock_blocks);
		// Remove this block from the list and release it.
		list_del_init(&va_block->uxu_lru);
		uvm_mutex_unlock(&uxu_va_space->lock_blocks);

		uvm_uxu_release_block(va_block);
		++counter;
	}

	uvm_va_space_up_write(va_space);

	return status;
}

/**
 * Write the data of this `va_block` to the file.
 * Callers have to make sure that there is no duplicated data on GPU.
 *
 * @param va_space: va_space that governs this operation.
 * @param va_range: UVM va_range.
 * @param va_block: the data source.
 * @param is_evict: indicate that this function is called do to eviction not flush.
 * @param page_mask: indicate which pages to be written out to the file. Ignore
 * if NULL.
 *
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
NV_STATUS
uvm_uxu_flush_host_block(uvm_va_space_t *va_space, uvm_va_range_t *va_range, uvm_va_block_t *va_block, bool is_evict, const uvm_page_mask_t *page_mask)
{
	NV_STATUS	status = NV_OK;

	struct file	*uxu_file = va_range->node.uxu_rtn.filp;
	mm_segment_t	fs;

	int	page_id, prev_page_id;

	// Compute the file start offset based on `va_block`.
	loff_t	file_start_offset = va_block->start - va_range->node.start;
	loff_t	offset;

	struct kiocb	kiocb;
	struct iovec	*iov = va_range->node.uxu_rtn.iov;
	struct iov_iter	iter;
	unsigned int	iov_index = 0;
	ssize_t	_ret;

	void	*page_addr;

	uvm_va_block_region_t	region = uvm_va_block_region(0, (va_block->end - va_block->start + 1) / PAGE_SIZE);

	uvm_page_mask_t	mask;

	UVM_ASSERT(uxu_file != NULL);

	if (!page_mask)
		uvm_page_mask_fill(&mask);
	else
		uvm_page_mask_copy(&mask, page_mask);

	// Switch the filesystem space to kernel space.
	fs = get_fs();
	set_fs(KERNEL_DS);

	// Build iov based on the page addresses.
	prev_page_id = -2;
	offset = file_start_offset;
	for_each_va_block_page_in_region_mask(page_id, &mask, region) {
		if (!va_block->cpu.pages[page_id])
			continue;

		page_addr = page_address(va_block->cpu.pages[page_id]);

		// Perform asynchronous write.
		if (page_id - 1 != prev_page_id && iov_index > 0) {
			init_sync_kiocb(&kiocb, uxu_file);
			kiocb.ki_pos = offset;
			iov_iter_init(&iter, WRITE, iov, iov_index, iov_index * PAGE_SIZE);
			_ret = call_write_iter(uxu_file, &kiocb, &iter);
			BUG_ON(_ret == -EIOCBQUEUED);

			iov_index = 0;
			offset = file_start_offset + page_id * PAGE_SIZE;
		}
		iov[iov_index].iov_base = page_addr;
		iov[iov_index].iov_len = PAGE_SIZE;
		++iov_index;
		prev_page_id = page_id;
	}

	// Start asynchronous write.
	if (iov_index > 0) {
		init_sync_kiocb(&kiocb, uxu_file);
		kiocb.ki_pos = offset;
		iov_iter_init(&iter, WRITE, iov, iov_index, iov_index * PAGE_SIZE);
		_ret = call_write_iter(uxu_file, &kiocb, &iter);
		BUG_ON(_ret == -EIOCBQUEUED);
	}

	// Mark that this block has dirty data on the file.
	va_block->is_dirty = true;

	// Switch back to the original space.
	set_fs(fs);

	return status;
}

void
uvm_uxu_set_page_dirty(struct page *page)
{
	/* Ugly, but we have no way to safely set dirty */
	int *p = (int *)page_to_virt(page);
	if (p) {
		int x;
		x = *p;
		*p = (x + 1);
		*p = x;
	}
}

NV_STATUS uvm_api_uxu_initialize(UVM_UXU_INITIALIZE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    return uvm_uxu_initialize(
        va_space,
        params->trash_nr_blocks,
        params->trash_reserved_nr_pages,
        params->flags
    );
}

NV_STATUS uvm_api_uxu_map(UVM_UXU_MAP_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    /* TODO: need check private data of dragon file */

    return uvm_uxu_map(va_space, params);
}

NV_STATUS uvm_api_uxu_remap(UVM_UXU_REMAP_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    return uvm_uxu_remap(va_space, params);
}
