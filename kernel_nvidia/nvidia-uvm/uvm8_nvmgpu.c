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
#include "uvm8_nvmgpu.h"

#define MIN(x,y) (x < y ? x : y)

static void *fsdata_array[PAGES_PER_UVM_VA_BLOCK];

/**
 * Initialize the NVMGPU module. This function has to be called once per
 * va_space. It must be called before calling
 * "uvm_nvmgpu_register_file_va_space"
 *
 * @param va_space: va_space to be initialized this module with.
 *
 * @param trash_nr_blocks: maximum number of va_block NVMGPU should evict out
 * at one time.
 *
 * @param trash_reserved_nr_pages: NVMGPU will automatically evicts va_block
 * when number of free pages plus number of page-cache pages less than this
 * value.
 *
 * @param flags: the flags that dictate the optimization behaviors. See
 * UVM_NVMGPU_INIT_* for more details.
 *
 * @return: NV_ERR_INVALID_OPERATION if `va_space` has been initialized already,
 * otherwise NV_OK.
 */
NV_STATUS uvm_nvmgpu_initialize(uvm_va_space_t *va_space, unsigned long trash_nr_blocks, unsigned long trash_reserved_nr_pages, unsigned short flags)
{
    uvm_nvmgpu_va_space_t *nvmgpu_va_space = &va_space->nvmgpu_va_space;

    if (!nvmgpu_va_space->is_initailized)
    {
        INIT_LIST_HEAD(&nvmgpu_va_space->lru_head);
	/* TODO: Lower down the locking order.
	 * Because invalid locking order warnings are generated when debug mode is enabled.
	 */
        uvm_mutex_init(&nvmgpu_va_space->lock, UVM_LOCK_ORDER_VA_SPACE);
        nvmgpu_va_space->trash_nr_blocks = trash_nr_blocks;
        nvmgpu_va_space->trash_reserved_nr_pages = trash_reserved_nr_pages;
        nvmgpu_va_space->flags = flags;
        nvmgpu_va_space->is_initailized = true;

        return NV_OK;
    }
    else
        return NV_ERR_INVALID_OPERATION;
}


/**
 * Register a file to this `va_space`.
 * NVMGPU will start tracking this UVM region if this function return success.
 *
 * @param va_space: va_space to register the file to.
 *
 * @param params: register parameters containing info about the file, size, etc.
 *
 * @return: NV_OK on success, NV_ERR_* otherwise.
 */
NV_STATUS uvm_nvmgpu_register_file_va_space(uvm_va_space_t *va_space, UVM_NVMGPU_REGISTER_FILE_VA_SPACE_PARAMS *params)
{
    NV_STATUS ret = NV_OK;
    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn;

    uvm_range_tree_node_t *node = uvm_range_tree_find(&va_space->va_range_tree, (NvU64)params->uvm_addr);
    NvU64 expected_start_addr = (NvU64)params->uvm_addr;
    NvU64 expected_end_addr = expected_start_addr + params->size - 1;

    size_t max_nr_blocks;

    // Make sure that uvm_nvmgpu_initialize is called before this function.
    if (!va_space->nvmgpu_va_space.is_initailized)
    {
        printk(KERN_DEBUG "Error: Call uvm_nvmgpu_register_file_va_space before uvm_nvmgpu_initialize\n");
        return NV_ERR_INVALID_OPERATION;
    }

    // Find uvm node associated with the specified UVM address. Might fail if
    // the library does not call cudaMallocaManaged before calling this
    // function.
    if (!node || node->start != expected_start_addr) {
        printk(KERN_DEBUG "Cannot find uvm range 0x%llx - 0x%llx\n", expected_start_addr, expected_end_addr);
        if (node)
            printk(KERN_DEBUG "Closet uvm range 0x%llx - 0x%llx\n", node->start, node->end);
        return NV_ERR_OPERATING_SYSTEM;
    }

    nvmgpu_rtn = &node->nvmgpu_rtn;

    // Get the struct file from the input file descriptor.
    if ((nvmgpu_rtn->filp = fget(params->backing_fd)) == NULL) {
        printk(KERN_DEBUG "Cannot find the backing fd: %d\n", params->backing_fd);
        return NV_ERR_OPERATING_SYSTEM;
    }

    // Record the flags and the file size.
    nvmgpu_rtn->flags = params->flags;
    nvmgpu_rtn->size = params->size;

    // Calculate the number of blocks associated with this UVM range.
    max_nr_blocks = uvm_va_range_num_blocks(container_of(node, uvm_va_range_t, node));

    // Allocate the bitmap to tell which blocks have dirty data on the file.
    nvmgpu_rtn->is_file_dirty_bitmaps = kzalloc(sizeof(unsigned long) * BITS_TO_LONGS(max_nr_blocks), GFP_KERNEL);
    if (!nvmgpu_rtn->is_file_dirty_bitmaps) {
        ret = NV_ERR_NO_MEMORY;
        goto _register_err_0;
    }

    // Allocate the bitmap to tell which blocks have data cached on the host.
    nvmgpu_rtn->has_data_bitmaps = kzalloc(sizeof(unsigned long) * BITS_TO_LONGS(max_nr_blocks), GFP_KERNEL);
    if (!nvmgpu_rtn->has_data_bitmaps) {
        ret = NV_ERR_NO_MEMORY;
        goto _register_err_1;
    }

    nvmgpu_rtn->iov = kmalloc(sizeof(struct iovec) * PAGES_PER_UVM_VA_BLOCK, GFP_KERNEL);
    if (!nvmgpu_rtn->iov) {
        ret = NV_ERR_NO_MEMORY;
        goto _register_err_2;
    }

    return NV_OK; 

    // Found an error. Free allocated memory before go out.
_register_err_2:
    kfree(nvmgpu_rtn->has_data_bitmaps);
_register_err_1:
    kfree(nvmgpu_rtn->is_file_dirty_bitmaps);
_register_err_0:
    return ret;
}

NV_STATUS uvm_nvmgpu_remap(uvm_va_space_t *va_space, UVM_NVMGPU_REMAP_PARAMS *params)
{
    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn;
    uvm_va_block_t *va_block, *va_block_next;
    uvm_nvmgpu_va_space_t *nvmgpu_va_space = &va_space->nvmgpu_va_space;

    uvm_va_range_t *va_range = uvm_va_range_find(va_space, (NvU64)params->uvm_addr);
    NvU64 expected_start_addr = (NvU64)params->uvm_addr;

    // Make sure that uvm_nvmgpu_initialize is called before this function.
    if (!va_space->nvmgpu_va_space.is_initailized)
    {
        printk(KERN_DEBUG "Error: Call uvm_nvmgpu_remap before uvm_nvmgpu_initialize\n");
        return NV_ERR_INVALID_OPERATION;
    }

    if (!va_range || va_range->node.start != expected_start_addr) {
        printk(KERN_DEBUG "Cannot find uvm whose address starts from 0x%llx\n", expected_start_addr);
        if (va_range)
            printk(KERN_DEBUG "Closet uvm range 0x%llx - 0x%llx\n", va_range->node.start, va_range->node.end);
        return NV_ERR_OPERATING_SYSTEM;
    }

    nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

    if (nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_VOLATILE)
        uvm_mutex_lock(&nvmgpu_va_space->lock);

    // Volatile data is simply discarded even though it has been remapped with non-volatile
    for_each_va_block_in_va_range_safe(va_range, va_block, va_block_next) {
        uvm_nvmgpu_block_clear_file_dirty(va_block);
        if (nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_VOLATILE) {
            uvm_nvmgpu_release_block(va_block);
            list_del(&va_block->nvmgpu_lru);
        }
    }

    if (nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_VOLATILE)
        uvm_mutex_unlock(&nvmgpu_va_space->lock);

    nvmgpu_rtn->flags = params->flags;

    return NV_OK;
}

/**
 * Unregister the specified va_range.
 * NVMGPU will stop tracking this `va_range` after this point.
 *
 * @param va_range: va_range to be untracked.
 *
 * @return: always NV_OK.
 */
NV_STATUS uvm_nvmgpu_unregister_va_range(uvm_va_range_t *va_range)
{
    struct file *filp;

    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

    filp = nvmgpu_rtn->filp;

    UVM_ASSERT(filp != NULL);

    if (nvmgpu_rtn->is_file_dirty_bitmaps)
        kfree(nvmgpu_rtn->is_file_dirty_bitmaps);

    if (nvmgpu_rtn->has_data_bitmaps)
        kfree(nvmgpu_rtn->has_data_bitmaps);

    if (nvmgpu_rtn->iov)
        kfree(nvmgpu_rtn->iov);

    if ((nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_WRITE) && !(nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_VOLATILE))
        vfs_fsync(filp, 1);

    fput(filp);

    return NV_OK;
}

static void uvm_nvmgpu_unmap_page(uvm_va_block_t *va_block, int page_index)
{
    uvm_gpu_id_t id;

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

static NV_STATUS insert_pagecache_to_va_block(uvm_va_block_t *va_block, int page_id, struct page *page)
{
    NV_STATUS status = NV_OK;
    uvm_gpu_id_t gpu_id;

    lock_page(page);

    if (va_block->cpu.pages[page_id] != page) {
        if (va_block->cpu.pages[page_id] != NULL) {
            uvm_nvmgpu_unmap_page(va_block, page_id);
            if (uvm_page_mask_test(&va_block->cpu.pagecached, page_id))
                put_page(page);
	    else
                __free_page(page);
	}
        for_each_gpu_id(gpu_id) {
            uvm_gpu_t *gpu;
            uvm_va_block_gpu_state_t *gpu_state = va_block->gpus[uvm_id_gpu_index(gpu_id)];
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
    uvm_nvmgpu_unmap_page(va_block, page_id);
    unlock_page(page);

    return status;
}

/**
 * Inspired by generic_file_buffered_read in /mm/filemap.c.
 */
static int prepare_page_for_read(struct file *filp, loff_t ppos, uvm_va_block_t *va_block, int page_id)
{
    struct address_space *mapping = filp->f_mapping;
    struct inode *inode = mapping->host;
    struct file_ra_state *ra = &filp->f_ra;
    pgoff_t index;
    pgoff_t last_index;
    pgoff_t prev_index;
    unsigned long offset;      /* offset into pagecache page */
    unsigned int prev_offset;
    int error = 0;

    index = ppos >> PAGE_SHIFT;
    prev_index = ra->prev_pos >> PAGE_SHIFT;
    prev_offset = ra->prev_pos & (PAGE_SIZE-1);
    last_index = (ppos + PAGE_SIZE + PAGE_SIZE-1) >> PAGE_SHIFT;
    offset = ppos & ~PAGE_MASK;

    for (;;) {
        struct page *page;
        pgoff_t end_index;
        loff_t isize;
        unsigned long nr;
        NV_STATUS ret;

        cond_resched();
find_page:
        if (fatal_signal_pending(current)) {
            error = -EINTR;
            goto out;
        }

        page = find_get_page(mapping, index);
        if (!page) {
            page_cache_sync_readahead(mapping,
                    ra, filp,
                    index, last_index - index);
            page = find_get_page(mapping, index);
            if (unlikely(page == NULL))
                goto no_cached_page;
        }
        if (PageReadahead(page)) {
            page_cache_async_readahead(mapping,
                    ra, filp, page,
                    index, last_index - index);
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
            if (!mapping->a_ops->is_partially_uptodate(page,
                        offset, PAGE_SIZE))
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

/**
 * copied from prepare_page_for_read. Ugly but keeping dragon codes as much as possble.
 */
static struct page *prepare_page_for_read2(struct file *filp, loff_t ppos, uvm_va_block_t *va_block, int page_id)
{
    struct address_space *mapping = filp->f_mapping;
    struct inode *inode = mapping->host;
    struct file_ra_state *ra = &filp->f_ra;
    pgoff_t index;
    pgoff_t last_index;
    pgoff_t prev_index;
    unsigned long offset;      /* offset into pagecache page */
    unsigned int prev_offset;
    struct page *page = NULL;
    int error = 0;

    index = ppos >> PAGE_SHIFT;
    prev_index = ra->prev_pos >> PAGE_SHIFT;
    prev_offset = ra->prev_pos & (PAGE_SIZE-1);
    last_index = (ppos + PAGE_SIZE + PAGE_SIZE-1) >> PAGE_SHIFT;
    offset = ppos & ~PAGE_MASK;

    for (;;) {
        pgoff_t end_index;
        loff_t isize;
        unsigned long nr;
        NV_STATUS ret;

        cond_resched();
find_page:
        if (fatal_signal_pending(current)) {
            error = -EINTR;
            goto out;
        }

        page = find_get_page(mapping, index);
        if (!page) {
            page_cache_sync_readahead(mapping,
                    ra, filp,
                    index, last_index - index);
            page = find_get_page(mapping, index);
            if (unlikely(page == NULL))
                goto no_cached_page;
        }
        if (PageReadahead(page)) {
            page_cache_async_readahead(mapping,
                    ra, filp, page,
                    index, last_index - index);
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
            if (!mapping->a_ops->is_partially_uptodate(page,
                        offset, PAGE_SIZE))
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
            page = NULL;
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
	unlock_page(page);

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
    return page;
}

struct page *
assign_pagecache(uvm_va_block_t *block, uvm_page_index_t page_index)
{
    uvm_va_range_t *va_range = block->va_range;
    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_range->node.nvmgpu_rtn;
    struct file *nvmgpu_file = nvmgpu_rtn->filp;
    loff_t file_start_offset = block->start - block->va_range->node.start;
    loff_t offset;
    int page_id = page_index;

    offset = file_start_offset + page_id * PAGE_SIZE;
    return prepare_page_for_read2(nvmgpu_file, offset, block, page_id);
}

static bool
fill_pagecaches_for_read(struct file *nvmgpu_file, uvm_va_block_t *va_block, uvm_va_block_region_t region)
{
    struct inode *inode = nvmgpu_file->f_mapping->host;
    loff_t isize;
    uvm_page_mask_t read_mask;
    int page_id;
    // Calculate the file offset based on the block start address.
    loff_t file_start_offset = va_block->start - va_block->va_range->node.start;

    uvm_page_mask_fill(&read_mask);

    isize = i_size_read(inode);

    // Fill in page-cache pages to va_block
    for_each_va_block_page_in_region_mask(page_id, &read_mask, region) {
        loff_t offset = file_start_offset + page_id * PAGE_SIZE;

        if (unlikely(offset >= isize)) {
            struct page	*page = va_block->cpu.pages[page_id];
            if (page)
                lock_page(page);
	    continue;
	}
        if (prepare_page_for_read(nvmgpu_file, offset, va_block, page_id) != 0) {
            printk(KERN_DEBUG "Cannot prepare page for read at file offset 0x%llx\n", offset);
	    return false;
        }
        UVM_ASSERT(va_block->cpu.pages[page_id]);
    }

    return true;
}

static uvm_page_index_t
get_region_readable_outer(uvm_va_block_t *va_block, struct file *nvmgpu_file)
{
    uvm_page_index_t outer = ((va_block->end - va_block->start) >> PAGE_SHIFT) + 1;
    uvm_page_index_t outer_max;
    struct inode *inode = nvmgpu_file->f_mapping->host;
    loff_t len_remain = i_size_read(inode) - (va_block->start - va_block->va_range->node.start);

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
NV_STATUS uvm_nvmgpu_read_begin(uvm_va_block_t *va_block, uvm_va_block_retry_t *block_retry, uvm_service_block_context_t *service_context)
{
    NV_STATUS status = NV_OK;

    uvm_va_range_t *va_range = va_block->va_range;

    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

    struct file *nvmgpu_file = nvmgpu_rtn->filp;

    // Specify that the entire block is the region of concern.
    uvm_va_block_region_t region = uvm_va_block_region(0, get_region_readable_outer(va_block, nvmgpu_file));

    uvm_page_mask_t my_mask;
    // Record the original page mask and set the mask to all 1s.
    uvm_page_mask_t original_page_mask;
    uvm_page_mask_copy(&original_page_mask, &service_context->block_context.make_resident.page_mask);

    uvm_page_mask_init_from_region(&service_context->block_context.make_resident.page_mask, region, NULL);
    uvm_page_mask_copy(&my_mask, &service_context->block_context.make_resident.page_mask);

    UVM_ASSERT(nvmgpu_file != NULL);

    if (!uvm_nvmgpu_block_has_data(va_block)) {
        bool is_file_dirty = uvm_nvmgpu_block_file_dirty(va_block);

        // Prevent block_populate_pages from allocating new pages
        uvm_nvmgpu_block_set_file_dirty(va_block);

        // Change this va_block's state: all pages are the residents of CPU.
        status = uvm_va_block_make_resident(va_block,
                                            block_retry,
                                            &service_context->block_context,
                                            UVM_ID_CPU,
                                            region,
                                            &my_mask,
                                            NULL,
                                            UVM_MAKE_RESIDENT_CAUSE_NVMGPU);

        // Return the dirty state to the original
        if (!is_file_dirty)
            uvm_nvmgpu_block_clear_file_dirty(va_block);

        if (status != NV_OK) {
            printk(KERN_DEBUG "Cannot make temporary resident on CPU\n");
            goto read_begin_err_0;
        }

        status = uvm_tracker_wait(&va_block->tracker);
        if (status != NV_OK) {
            printk(KERN_DEBUG "Cannot make temporary resident on CPU\n");
            goto read_begin_err_0;
        }
    }

    if (fill_pagecaches_for_read(nvmgpu_file, va_block, region)) {
        uvm_nvmgpu_block_set_has_data(va_block);
    }
    else {
        status = NV_ERR_OPERATING_SYSTEM;
    }

read_begin_err_0:
    // Put back the original mask.
    uvm_page_mask_copy(&service_context->block_context.make_resident.page_mask, &original_page_mask);
    
    return status;
}

NV_STATUS uvm_nvmgpu_read_end(uvm_va_block_t *va_block)
{
    int page_id;
    struct page *page;

    uvm_page_mask_t read_mask;

    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_block->va_range->node.nvmgpu_rtn;
    struct file *nvmgpu_file = nvmgpu_rtn->filp;
    uvm_va_block_region_t region = uvm_va_block_region(0, get_region_readable_outer(va_block, nvmgpu_file));

    uvm_page_mask_fill(&read_mask);
    for_each_va_block_page_in_region_mask(page_id, &read_mask, region) {
        page = va_block->cpu.pages[page_id];
        if (page)
	    unlock_page(page);
    }

    return NV_OK;
}

/**
 * Evict out the block. This function can handle both CPU-only and GPU blocks.
 * 
 * @param va_block: the block to be evicted.
 * 
 * @return: NV_OK on success. NV_ERR_* otherwise.
 */
NV_STATUS uvm_nvmgpu_flush_block(uvm_va_block_t *va_block)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_t *va_range = va_block->va_range;
    uvm_va_space_t *va_space = va_range->va_space;
    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_range->node.nvmgpu_rtn;

    if (!(nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_WRITE))
        return NV_OK;

    // Move data from GPU to CPU
    if (uvm_processor_mask_get_gpu_count(&(va_block->resident)) > 0) {
        uvm_va_block_region_t region = uvm_va_block_region_from_block(va_block);
        uvm_va_block_context_t *block_context = uvm_va_block_context_alloc();

        if (!block_context) {
            printk(KERN_DEBUG "NV_ERR_NO_MEMORY\n");
            return NV_ERR_NO_MEMORY;
        }

        // Force direct flush into the file for UVM_NVMGPU_FLAG_USEHOSTBUF that has no host buffer
        if ((nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_USEHOSTBUF) 
            && !va_block->nvmgpu_use_uvm_buffer
        )
            uvm_nvmgpu_block_set_file_dirty(va_block);

	uvm_mutex_lock(&va_block->lock);
        // Move data resided on the GPU to host.
        // Data is automatically moved to the file if UVM_NVMGPU_FLAG_USEHOSTBUF is unset.
        status = uvm_va_block_migrate_locked(
            va_block, 
            NULL, 
            block_context, 
            region, 
            UVM_ID_CPU, 
            UVM_MIGRATE_MODE_MAKE_RESIDENT, 
            NULL
        );
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

    // Flush the data kept in the host memory
    if ((nvmgpu_rtn->flags & UVM_NVMGPU_FLAG_USEHOSTBUF)
        && va_block->nvmgpu_use_uvm_buffer
    ) {
        status = uvm_nvmgpu_flush_host_block(va_space, va_range, va_block, false, NULL);
        if (status != NV_OK) {
            printk(KERN_DEBUG "CANNOT FLUSH HOST BLOCK\n");
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
NV_STATUS uvm_nvmgpu_flush(uvm_va_range_t *va_range)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_t *va_block, *va_block_next;

    // Evict blocks one by one.
    for_each_va_block_in_va_range_safe(va_range, va_block, va_block_next) {
        if ((status = uvm_nvmgpu_flush_block(va_block)) != NV_OK) {
            printk(KERN_DEBUG "Encountered a problem with uvm_nvmgpu_flush_block\n");
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
NV_STATUS uvm_nvmgpu_release_block(uvm_va_block_t *va_block)
{
    uvm_va_block_t *old;
    size_t index;
    
    uvm_va_range_t *va_range = va_block->va_range;

    UVM_ASSERT(va_block != NULL);

    // Remove the block from the list.
    index = uvm_va_range_block_index(va_range, va_block->start);
    old = (uvm_va_block_t *)nv_atomic_long_cmpxchg(&va_range->blocks[index],
                                                  (long)va_block,
                                                  (long)NULL);

    // Free the block.
    if (old == va_block) {
        uvm_nvmgpu_block_clear_has_data(va_block);
        uvm_va_block_kill(va_block);
    }

    return NV_OK;
}

NV_STATUS uvm_nvmgpu_prepare_block_for_hostbuf(uvm_va_block_t *va_block)
{
    int page_id;
    if (!va_block->nvmgpu_use_uvm_buffer) {
        for (page_id = 0; page_id < PAGES_PER_UVM_VA_BLOCK; ++page_id) {
            if (va_block->cpu.pages[page_id] != NULL) {
                uvm_nvmgpu_unmap_page(va_block, page_id);
                va_block->cpu.pages[page_id] = NULL;
            }
        }
    }
    return NV_OK;
}

NV_STATUS uvm_nvmgpu_write_begin(uvm_va_block_t *va_block, bool is_flush)
{
    NV_STATUS status = NV_OK;

    int page_id;
    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_block->va_range->node.nvmgpu_rtn;

    // Calculate the file offset based on the block start address.
    loff_t file_start_offset = va_block->start - va_block->va_range->node.start;
    loff_t file_position;

    struct file *nvmgpu_file = nvmgpu_rtn->filp;
    struct inode *f_inode = file_inode(nvmgpu_file);
    struct address_space *mapping = nvmgpu_file->f_mapping;
    struct inode *m_inode = mapping->host;
    const struct address_space_operations *a_ops = mapping->a_ops;

    struct page *page;
    void *fsdata;

    uvm_va_space_t *va_space;

    UVM_ASSERT(va_block->va_range);
    UVM_ASSERT(va_block->va_range->va_space);
    va_space = va_block->va_range->va_space;

    inode_lock(f_inode);

    current->backing_dev_info = inode_to_bdi(m_inode);

    file_remove_privs(nvmgpu_file);

    file_update_time(nvmgpu_file);

    for (page_id = 0; page_id < PAGES_PER_UVM_VA_BLOCK; ++page_id) {
        uvm_gpu_id_t id;
        long f_status = 0;

        file_position = file_start_offset + page_id * PAGE_SIZE;

        if (file_position >= nvmgpu_rtn->size)
            break;

        f_status = a_ops->write_begin(
            nvmgpu_file, 
            mapping, 
            file_position, 
            MIN(PAGE_SIZE, nvmgpu_rtn->size - file_position), 
            0, 
            &page, 
            &fsdata
        );
        
        if (f_status != 0 || page == NULL)
            continue;

        if (mapping_writably_mapped(mapping))
            flush_dcache_page(page);

        fsdata_array[page_id] = fsdata;

        if (va_block->cpu.pages[page_id] != NULL)
            uvm_nvmgpu_unmap_page(va_block, page_id);

        for_each_gpu_id(id) {
            uvm_gpu_t *gpu;
            uvm_va_block_gpu_state_t *gpu_state = va_block->gpus[uvm_id_gpu_index(id)];
            if (!gpu_state)
                continue;

            UVM_ASSERT(gpu_state->cpu_pages_dma_addrs[page_id] == 0);

            gpu = uvm_va_space_get_gpu(va_space, id);

            status = uvm_gpu_map_cpu_pages(gpu, page, PAGE_SIZE, &gpu_state->cpu_pages_dma_addrs[page_id]);
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

NV_STATUS uvm_nvmgpu_write_end(uvm_va_block_t *va_block, bool is_flush)
{
    NV_STATUS status = NV_OK;

    uvm_nvmgpu_range_tree_node_t *nvmgpu_rtn = &va_block->va_range->node.nvmgpu_rtn;
    struct file *nvmgpu_file = nvmgpu_rtn->filp;
    struct inode *f_inode = file_inode(nvmgpu_file);
    struct address_space *mapping = nvmgpu_file->f_mapping;
    const struct address_space_operations *a_ops = mapping->a_ops;

    int page_id;

    loff_t file_start_offset = va_block->start - va_block->va_range->node.start;
    loff_t file_position;

    for (page_id = 0; page_id < PAGES_PER_UVM_VA_BLOCK; ++page_id) {
        struct page *page = va_block->cpu.pages[page_id];
        void *fsdata = fsdata_array[page_id];

        file_position = file_start_offset + page_id * PAGE_SIZE;

        if (file_position >= nvmgpu_rtn->size)
            break;

        if (page) {
            size_t bytes = MIN(PAGE_SIZE, nvmgpu_rtn->size - file_position);
            flush_dcache_page(page);
            mark_page_accessed(page);

            a_ops->write_end(
                nvmgpu_file, 
                mapping, 
                file_position, 
                bytes, 
                bytes, 
                page, 
                fsdata
            );

            balance_dirty_pages_ratelimited(mapping);
        }
    }

    uvm_nvmgpu_block_set_has_data(va_block);
    uvm_nvmgpu_block_set_file_dirty(va_block);

    current->backing_dev_info = NULL;

    inode_unlock(f_inode);

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
NV_STATUS uvm_nvmgpu_reduce_memory_consumption(uvm_va_space_t *va_space)
{
   /*
    * TODO: locking assertion failed. write lock is required.
    */
    NV_STATUS status = NV_OK;

    uvm_nvmgpu_va_space_t *nvmgpu_va_space = &va_space->nvmgpu_va_space;

    unsigned long counter = 0;

    uvm_va_block_t *va_block;

    uvm_va_space_down_write(va_space);
    // Reclaim blocks based on least recent transfer.

    while (!list_empty(&nvmgpu_va_space->lru_head) && counter < nvmgpu_va_space->trash_nr_blocks) {
        va_block = list_first_entry(&nvmgpu_va_space->lru_head, uvm_va_block_t, nvmgpu_lru);

        // Terminate the loop since we cannot trash out blocks that have a copy on GPU
        if (uvm_processor_mask_get_gpu_count(&(va_block->resident)) > 0) {
            printk(KERN_DEBUG "Encounter a block whose data are in GPU!!!\n");
            break;
        }

        // Evict the block if it is on CPU only and this `va_range` has the write flag.
        if (uvm_processor_mask_get_count(&(va_block->resident)) > 0 && va_block->va_range->node.nvmgpu_rtn.flags & UVM_NVMGPU_FLAG_WRITE) {
            status = uvm_nvmgpu_flush_host_block(va_block->va_range->va_space, va_block->va_range, va_block, true, NULL);
            if (status != NV_OK) {
                printk(KERN_DEBUG "Cannot evict block\n");
                break;
            }
        }

        // Remove this block from the list and release it.
        list_del(nvmgpu_va_space->lru_head.next);
        uvm_nvmgpu_release_block(va_block);
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
NV_STATUS uvm_nvmgpu_flush_host_block(uvm_va_space_t *va_space, uvm_va_range_t *va_range, uvm_va_block_t *va_block, bool is_evict, const uvm_page_mask_t *page_mask)
{
    NV_STATUS status = NV_OK;

    struct file *nvmgpu_file = va_range->node.nvmgpu_rtn.filp;
    mm_segment_t fs;

    int page_id, prev_page_id;

    // Compute the file start offset based on `va_block`.
    loff_t file_start_offset = va_block->start - va_range->node.start;
    loff_t offset;

    struct kiocb kiocb;
    struct iovec *iov = va_range->node.nvmgpu_rtn.iov;
    struct iov_iter iter;
    unsigned int iov_index = 0;
    ssize_t _ret;

    void *page_addr;

    uvm_va_block_region_t region = uvm_va_block_region(0, (va_block->end - va_block->start + 1) / PAGE_SIZE);

    uvm_page_mask_t mask;

    UVM_ASSERT(nvmgpu_file != NULL);

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
            init_sync_kiocb(&kiocb, nvmgpu_file);
            kiocb.ki_pos = offset;
            iov_iter_init(&iter, WRITE, iov, iov_index, iov_index * PAGE_SIZE);
            _ret = call_write_iter(nvmgpu_file, &kiocb, &iter);
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
        init_sync_kiocb(&kiocb, nvmgpu_file);
        kiocb.ki_pos = offset;
        iov_iter_init(&iter, WRITE, iov, iov_index, iov_index * PAGE_SIZE);
        _ret = call_write_iter(nvmgpu_file, &kiocb, &iter);
        BUG_ON(_ret == -EIOCBQUEUED);
    }
    
    // Mark that this block has dirty data on the file.
    uvm_nvmgpu_block_set_file_dirty(va_block);

    // Switch back to the original space.
    set_fs(fs);

    return status;
}
