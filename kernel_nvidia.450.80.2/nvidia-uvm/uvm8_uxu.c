#include <linux/syscalls.h>
#include <linux/delay.h>
#include <linux/aio.h>
#include <linux/swap.h>
#include <linux/writeback.h>
#include <linux/fs.h>
#include <linux/backing-dev.h>
#include <linux/uio.h>
#include <linux/buffer_head.h>

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

#define UXU_FILE_FROM_RANGE(range)	((range)->node.uxu_rtn.filp)
#define UXU_FILE_FROM_BLOCK(block)	UXU_FILE_FROM_RANGE((block)->va_range)
#define BLOCK_START_OFFSET(block)	((block)->start - (block)->va_range->node.start)

struct proc_dir_entry	*procfs_entry_uxu;

static atomic64_t	n_uxu_blks;

/**
 * Determine if we need to reclaim some blocks or not.
 *
 * @param uxu_va_space: the va_space information related to UXU.
 *
 * @return: true if we need to reclaim, false otherwise.
 */
static inline bool
uxu_has_to_reclaim_blocks(uvm_uxu_va_space_t *uxu_va_space)
{
	unsigned long	freeram = global_zone_page_state(NR_FREE_PAGES);
	unsigned long	pagecacheram = global_zone_page_state(NR_FILE_PAGES);
	return freeram + pagecacheram < uxu_va_space->reserved_nr_pages;
}

/**
 * Mark that we just touch this block, which has in-buffer data.
 *
 * @param va_block: va_block to be marked.
 */
static inline void
uxu_block_mark_recent_in_buffer(uvm_va_block_t *va_block)
{
	uvm_uxu_va_space_t *uxu_va_space = &va_block->va_range->va_space->uxu_va_space;

	// Move this block to the tail of the LRU list.
	// mutex locking is commented out. It has been already held.
        uvm_mutex_lock(&uxu_va_space->lock_blocks);
	if (!list_empty(&va_block->uxu_lru))
		list_move_tail(&va_block->uxu_lru, &uxu_va_space->lru_head);
        uvm_mutex_unlock(&uxu_va_space->lock_blocks);
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

static void
uxu_unmap_page(uvm_va_block_t *va_block, int page_index)
{
	uvm_gpu_id_t	id;

	for_each_gpu_id(id) {
		uvm_gpu_t	*gpu;
		uvm_va_block_gpu_state_t	*gpu_state = va_block->gpus[uvm_id_gpu_index(id)];

		if (!gpu_state)
			continue;

		if (gpu_state->cpu_pages_dma_addrs[page_index] == 0)
			continue;

		gpu = uvm_va_space_get_gpu(va_block->va_range->va_space, id);

		uvm_gpu_unmap_cpu_page(gpu, gpu_state->cpu_pages_dma_addrs[page_index]);
		gpu_state->cpu_pages_dma_addrs[page_index] = 0;
	}
}

static NV_STATUS
add_pagecache_to_block(uvm_va_block_t *block, int page_id, struct page *page)
{
	uvm_gpu_id_t	gpu_id;

	UVM_ASSERT(block->cpu.pages[page_id] == NULL);

	for_each_gpu_id(gpu_id) {
		uvm_gpu_t	*gpu;
		uvm_va_block_gpu_state_t	*gpu_state = block->gpus[uvm_id_gpu_index(gpu_id)];
		NV_STATUS	status;

		if (!gpu_state)
			continue;

		UVM_ASSERT(gpu_state->cpu_pages_dma_addrs[page_id] == 0);

		gpu = uvm_va_space_get_gpu(block->va_range->va_space, gpu_id);

		status = uvm_gpu_map_cpu_pages(gpu, page, PAGE_SIZE, &gpu_state->cpu_pages_dma_addrs[page_id]);
		if (status != NV_OK) {
			printk(KERN_DEBUG "Cannot do uvm_gpu_map_cpu_pages\n");

			uxu_unmap_page(block, page_id);

			return status;
		}
	}
	block->cpu.pages[page_id] = page;
	return NV_OK;
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
	struct file	*uxu_file;
	pgoff_t	pgoff_block;
	struct page	*page;

	uxu_file = UXU_FILE_FROM_BLOCK(block);
	pgoff_block = BLOCK_START_OFFSET(block) >> PAGE_SHIFT;
	page = read_mapping_page(uxu_file->f_mapping, pgoff_block + page_index, NULL);
	if (IS_ERR(page))
		return NULL;
	return page;
}

static inline bool
uxu_is_pagecachable(uvm_va_block_t *block, uvm_page_index_t page_id)
{
	struct inode *inode;
	uvm_page_index_t	outer_max;
	loff_t len_remain;

	if (uxu_is_volatile_block(block))
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
		if (!IS_ERR(page)) {
			uvm_page_mask_set(&block->cpu.pagecached, page_index);
			if (!page_has_buffers(page)) {
				create_empty_buffers(page, block->va_range->node.uxu_rtn.filp->f_mapping->host->i_sb->s_blocksize, BIT(BH_Dirty) | BIT(BH_Uptodate));
			}
			/* TODO: It seems to be natural that marking a mapped page as dirty. */
			SetPageDirty(page);
		}
		else {
			page = NULL;
		}
	}
	else {
		page = assign_page(block, zero);
		if (page)
			uvm_page_mask_clear(&block->cpu.pagecached, page_index);
	}

	return page;
}

static void
setup_block_readable_region(uvm_va_block_t *block, uvm_va_block_region_t *pregion)
{
	struct file	*uxu_file;
	struct inode	*inode;
	uvm_page_index_t	outer, outer_readable;
	loff_t	len_remain;

	uxu_file = UXU_FILE_FROM_BLOCK(block);
	inode = uxu_file->f_mapping->host;
	len_remain = i_size_read(inode) - (block->start - block->va_range->node.start);

	outer = ((block->end - block->start) >> PAGE_SHIFT) + 1;
	outer_readable = (len_remain + PAGE_SIZE - 1) >> PAGE_SHIFT;
	if (outer > outer_readable)
		pregion->outer = outer_readable;
	else
		pregion->outer = outer;
	pregion->first = 0;
}

static bool
load_pagecaches_for_block(uvm_va_block_t *block)
{
	uvm_va_block_region_t	region;
	int	page_id;

	setup_block_readable_region(block, &region);
	// Fill in page-cache pages to va_block
	for_each_va_block_page_in_region(page_id, region) {
		struct page	*page;
		int	ret;

		page = uxu_get_page(block, page_id, false);
		if (page == NULL) {
			printk(KERN_DEBUG "failed to assign pagecache(block: %llx, page_id: %d\n", block->start, page_id);
			return false;
		}
		ret = add_pagecache_to_block(block, page_id, page);
		if (ret != NV_OK)
			put_page(page);
		uvm_page_mask_set(&block->cpu.resident, page_id);
	}

	uvm_processor_mask_set(&block->resident, UVM_ID_CPU);
	return true;
}

void
uxu_try_load_block(uvm_va_block_t *block, uvm_va_block_retry_t *block_retry, uvm_service_block_context_t *service_context, uvm_processor_id_t processor_id)
{
	if (block->is_loaded)
		return;
	if (uxu_is_volatile_block(block))
		return;
	if (uxu_is_read_block(block)) {
		if (!load_pagecaches_for_block(block))
			return;
	}

	block->is_loaded = TRUE;
	uxu_block_mark_recent_in_buffer(block);
}

/**
 * Evict out the block. This function can handle both CPU-only and GPU blocks.
 *
 * @param va_block: the block to be evicted.
 *
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
static NV_STATUS
uxu_flush_block(uvm_va_block_t *block)
{
	if (!uxu_is_write_block(block))
		return NV_OK;

	// Move data from GPU to CPU
	if (uvm_processor_mask_get_gpu_count(&block->resident) > 0) {
		uvm_va_block_region_t	region = uvm_va_block_region_from_block(block);
		uvm_va_block_context_t	*block_context = uvm_va_block_context_alloc();
		NV_STATUS	status;

		if (!block_context) {
			printk(KERN_DEBUG "NV_ERR_NO_MEMORY\n");
			return NV_ERR_NO_MEMORY;
		}

		uvm_mutex_lock(&block->lock);
		// Move data resided on the GPU to host.
		status = uvm_va_block_migrate_locked(block, NULL, block_context, region, UVM_ID_CPU, UVM_MIGRATE_MODE_MAKE_RESIDENT, NULL);
		uvm_mutex_unlock(&block->lock);

		uvm_va_block_context_free(block_context);

		if (status != NV_OK) {
			printk(KERN_DEBUG "NOT NV_OK\n");
			return status;
		}

		// Wait for the d2h transfer to complete.
		status = uvm_tracker_wait(&block->tracker);

		if (status != NV_OK) {
			printk(KERN_DEBUG "NOT NV_OK\n");
			return status;
		}
	}

	return NV_OK;
}

/**
 * Flush all blocks in the `va_range`.
 *
 * @param va_range: va_range that we want to flush the data.
 *
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
static NV_STATUS
uxu_flush(uvm_va_range_t *va_range)
{
	NV_STATUS	status = NV_OK;
	uvm_va_block_t	*block, *block_next;

	// Evict blocks one by one.
	for_each_va_block_in_va_range_safe(va_range, block, block_next) {
		if ((status = uxu_flush_block(block)) != NV_OK) {
			printk(KERN_DEBUG "Encountered a problem with uxu_flush_block\n");
			break;
		}
	}

	return status;
}

void
uxu_block_created(uvm_va_range_t *range, uvm_va_block_t *block)
{
	INIT_LIST_HEAD(&block->uxu_lru);
	if (uvm_is_uxu_range(range)) {
		uvm_uxu_va_space_t	*uxu_va_space = &range->va_space->uxu_va_space;

                uvm_mutex_lock(&uxu_va_space->lock_blocks);
                list_move_tail(&block->uxu_lru, &uxu_va_space->lru_head);
                uvm_mutex_unlock(&uxu_va_space->lock_blocks);
		atomic64_inc(&n_uxu_blks);
	}
}

/**
 * Unregister the specified va_range.
 * UXU will stop tracking this `va_range` after this point.
 *
 * @param range: va range to be being destroyed
 *
 */
void
uxu_range_destroyed(uvm_va_range_t *range)
{
	struct file	*filp = UXU_FILE_FROM_RANGE(range);

	if (uxu_is_write_range(range) && !uxu_is_volatile_range(range)) {
		uxu_flush(range);
		vfs_fsync(filp, 1);
	}

	fput(filp);

	if (range->blocks) {
		uvm_uxu_va_space_t	*uxu_va_space = &range->va_space->uxu_va_space;
		uvm_va_block_t	*block, *block_tmp;
		int	n_blks = 0;

                uvm_mutex_lock(&uxu_va_space->lock_blocks);
		for_each_va_block_in_va_range_safe(range, block, block_tmp) {
			list_del_init(&block->uxu_lru);
			n_blks++;
		}
                uvm_mutex_unlock(&uxu_va_space->lock_blocks);

		atomic64_sub(n_blks, &n_uxu_blks);
	}
}

/**
 * Free memory associated with the `va_block`.
 *
 * @param va_block: va_block to be freed.
 *
 * @return: always NV_OK;
 */
static NV_STATUS
uxu_release_block(uvm_va_block_t *block)
{
	uvm_va_block_t	*old;
	size_t	index;
	uvm_va_range_t	*range = block->va_range;

	if (range == NULL) {
		/* maybe already destroyed ?? */
		return NV_OK;
	}

	// Remove the block from the list.
	index = uvm_va_range_block_index(range, block->start);
	old = (uvm_va_block_t *)nv_atomic_long_cmpxchg(&range->blocks[index], (long)block, (long)NULL);

	// Free the block.
	if (old == block) {
		uvm_uxu_va_space_t	*uxu_va_space = &range->va_space->uxu_va_space;

		uvm_mutex_lock(&uxu_va_space->lock_blocks);
		list_del_init(&block->uxu_lru);
		uvm_mutex_unlock(&uxu_va_space->lock_blocks);

		atomic64_dec(&n_uxu_blks);

		uvm_va_block_kill(block);
	}

	return NV_OK;
}

/**
 * Automatically reduce memory usage if we need to.
 *
 * @param va_space: va_space that governs this operation.
 * @param force: if true, we will evict some blocks without checking for the memory pressure.
 *
 * @return: NV_OK on success, NV_ERR_* otherwise.
 */
static void
uxu_reduce_memory_consumption(uvm_va_space_t *va_space)
{
	/*
	 * TODO: locking assertion failed. write lock is required.
	 */
	uvm_uxu_va_space_t	*uxu_va_space = &va_space->uxu_va_space;
	struct list_head	*lp, *next;
	unsigned long	n_swapped;

	uvm_va_space_down_write(va_space);

	// Reclaim blocks based on least recent transfer.

	n_swapped = 0;
	list_for_each_safe(lp, next, &uxu_va_space->lru_head) {
		uvm_va_block_t	*block;

		if (n_swapped >= uxu_va_space->swapout_nr_blocks)
			break;
		block = list_entry(lp, uvm_va_block_t, uxu_lru);

		// Terminate the loop since we cannot swap out blocks that have a copy on GPU
		if (uvm_processor_mask_get_gpu_count(&block->resident) > 0) {
			// Encounter a block whose data are in GPU
			continue;
		}

		uxu_release_block(block);
		n_swapped++;
	}

	uvm_va_space_up_write(va_space);
}

static int
pagecache_reducer(void *ctx)
{
	uvm_va_space_t	*va_space = (uvm_va_space_t *)ctx;
	uvm_uxu_va_space_t	*uxu_va_space = &va_space->uxu_va_space;
	uvm_thread_context_wrapper_t	thread_context;

	uvm_thread_context_add(&thread_context.context);

	while (!kthread_should_stop()) {
		if (uxu_has_to_reclaim_blocks(uxu_va_space))
			uxu_reduce_memory_consumption(va_space);
		schedule_timeout_idle(10);
	}

	uvm_thread_context_remove(&thread_context.context);
	return 0;
}

/**
 * Initialize the UXU module. This function has to be called once per
 * va_space. It must be called before calling
 * "uvm_uxu_register_file_va_space"
 *
 * @param va_space: va_space to be initialized this module with.
 *
 * @param swapout_nr_blocks: maximum number of va_block UXU should evict out
 * at one time.
 *
 * @param reserved_nr_pages: UXU will automatically evicts va_block
 * when number of free pages plus number of page-cache pages less than this
 * value.
 *
 * @param flags: the flags that dictate the optimization behaviors. See
 * UVM_UXU_INIT_* for more details.
 *
 * @return: NV_ERR_INVALID_OPERATION if `va_space` has been initialized already,
 * otherwise NV_OK.
 */
static NV_STATUS
uxu_initialize(uvm_va_space_t *va_space, unsigned long swapout_nr_blocks, unsigned long reserved_nr_pages, unsigned short flags)
{
	uvm_uxu_va_space_t	*uxu_va_space = &va_space->uxu_va_space;

	if (!uxu_va_space->is_initailized) {
		INIT_LIST_HEAD(&uxu_va_space->lru_head);
		/* TODO: Lower down the locking order.
		 * Because invalid locking order warnings are generated when debug mode is enabled.
		 */
		uvm_mutex_init(&uxu_va_space->lock, UVM_LOCK_ORDER_VA_SPACE);
		uvm_mutex_init(&uxu_va_space->lock_blocks, UVM_LOCK_ORDER_VA_SPACE_UXU);
		uxu_va_space->swapout_nr_blocks = swapout_nr_blocks;
		uxu_va_space->reserved_nr_pages = reserved_nr_pages;
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
static NV_STATUS
uxu_map(uvm_va_space_t *va_space, UVM_UXU_MAP_PARAMS *params)
{
	uvm_uxu_range_tree_node_t	*uxu_rtn;
	uvm_range_tree_node_t	*node = uvm_range_tree_find(&va_space->va_range_tree, (NvU64)params->uvm_addr);
	NvU64	expected_start_addr = (NvU64)params->uvm_addr;
	NvU64	expected_end_addr = expected_start_addr + params->size - 1;
	size_t	max_nr_blocks;

	// Make sure that uxu_initialize is called before this function.
	if (!va_space->uxu_va_space.is_initailized) {
		printk(KERN_DEBUG "Error: Call uxu_register_file_va_space before uvm_uxu_initialize\n");
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

	return NV_OK;
}

static NV_STATUS
uxu_remap(uvm_va_space_t *va_space, UVM_UXU_REMAP_PARAMS *params)
{
	uvm_va_range_t	*va_range = uvm_va_range_find(va_space, (NvU64)params->uvm_addr);
	uvm_va_block_t	*block, *block_next;
	NvU64	expected_start_addr = (NvU64)params->uvm_addr;

	// Make sure that uxu_initialize is called before this function.
	if (!va_space->uxu_va_space.is_initailized) {
		printk(KERN_DEBUG "Error: Call uxu_remap before uxu_initialize\n");
		return NV_ERR_INVALID_OPERATION;
	}

	if (!va_range || va_range->node.start != expected_start_addr) {
		printk(KERN_DEBUG "Cannot find uvm whose address starts from 0x%llx\n", expected_start_addr);
		if (va_range)
			printk(KERN_DEBUG "Closet uvm range 0x%llx - 0x%llx\n", va_range->node.start, va_range->node.end);
		return NV_ERR_OPERATING_SYSTEM;
	}

	if (uxu_is_write_block(block)) {
		// Volatile data is simply discarded even though it has been remapped with non-volatile
		for_each_va_block_in_va_range_safe(va_range, block, block_next) {
			block->is_dirty = false;
			uxu_release_block(block);
		}
	}
	else {
		for_each_va_block_in_va_range_safe(va_range, block, block_next)
			block->is_dirty = false;
	}

	va_range->node.uxu_rtn.flags = params->flags;

	return NV_OK;
}

NV_STATUS
uvm_api_uxu_initialize(UVM_UXU_INITIALIZE_PARAMS *params, struct file *filp)
{
	uvm_va_space_t *va_space = uvm_va_space_get(filp);
	return uxu_initialize(va_space, params->swapout_nr_blocks, params->reserved_nr_pages, params->flags);
}

NV_STATUS
uvm_api_uxu_map(UVM_UXU_MAP_PARAMS *params, struct file *filp)
{
	uvm_va_space_t *va_space = uvm_va_space_get(filp);

	/* TODO: need check private data of dragon file */

	return uxu_map(va_space, params);
}

NV_STATUS
uvm_api_uxu_remap(UVM_UXU_REMAP_PARAMS *params, struct file *filp)
{
	uvm_va_space_t *va_space = uvm_va_space_get(filp);
	return uxu_remap(va_space, params);
}

static int
nv_procfs_read_uxu_stats(struct seq_file *s, void *v)
{
	if (!uvm_down_read_trylock(&g_uvm_global.pm.lock))
		return -EAGAIN;

	UVM_SEQ_OR_DBG_PRINT(s, "cezanne     %llu\n", (NvU64)atomic64_read(&n_uxu_blks));

	uvm_up_read(&g_uvm_global.pm.lock);

	return 0;
}

static int
nv_procfs_read_uxu_stats_entry(struct seq_file *s, void *v)
{
	UVM_ENTRY_RET(nv_procfs_read_uxu_stats(s, v));
}

UVM_DEFINE_SINGLE_PROCFS_FILE(uxu_stats_entry);

#define UXU_STATS_PROC_ENTRY_NAME	"uxu_stats"

NV_STATUS
uxu_init(void)
{
	struct proc_dir_entry	*cpu_base_dir_entry = uvm_procfs_get_cpu_base_dir();

        procfs_entry_uxu = NV_CREATE_PROC_FILE(UXU_STATS_PROC_ENTRY_NAME, cpu_base_dir_entry, uxu_stats_entry, NULL);
        if (procfs_entry_uxu == NULL)
		return NV_ERR_OPERATING_SYSTEM;
	return NV_OK;
}

void
uxu_exit(void)
{
	uvm_procfs_destroy_entry(procfs_entry_uxu);
}
