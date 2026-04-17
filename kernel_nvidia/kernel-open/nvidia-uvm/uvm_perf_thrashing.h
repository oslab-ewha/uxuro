/*******************************************************************************
    Copyright (c) 2016-2022 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM_PERF_THRASHING_H__
#define __UVM_PERF_THRASHING_H__

#include "uvm_linux.h"
#include "uvm_extern_decl.h"
#include "uvm_forward_decl.h"
#include "uvm_processors.h"
#include "uvm_va_block_types.h"

typedef enum
{
    // No thrashing detected
    UVM_PERF_THRASHING_HINT_TYPE_NONE     = 0,

    // Map remotely to avoid future faults (does not help with revocations due
    // to system-wide atomics)
    UVM_PERF_THRASHING_HINT_TYPE_PIN      = 1,

    // Throttle execution of the calling processor (this can be implemented by
    // sleeping or handing other faults)
    UVM_PERF_THRASHING_HINT_TYPE_THROTTLE = 2,

    // TODO: Bug 1877578: Implement heuristics-driven read-duplication
    // Add a thrashing hint type to read-duplicate a page when it is being
    // accessed read-only from different processors
} uvm_perf_thrashing_hint_type_t;

typedef struct
{
    uvm_perf_thrashing_hint_type_t type;

    union
    {
        struct
        {
            // Map to this processor, which must be accessible, at least, from
            // the calling processor
            uvm_processor_id_t residency;

            // Processors to be mapped, when possible, to the new residency
            uvm_processor_mask_t processors;
        } pin;

        struct
        {
            // Absolute timestamp in ns after which the throttled processor is
            // allowed to start servicing faults on the thrashing page.
            NvU64 end_time_stamp;
        } throttle;
    };
} uvm_perf_thrashing_hint_t;

// Obtain a hint to prevent thrashing on the page with given address
uvm_perf_thrashing_hint_t uvm_perf_thrashing_get_hint(uvm_va_block_t *va_block, NvU64 address,
                                                      uvm_processor_id_t requester);

// Obtain a pointer to a mask with the processors that are thrashing on the
// given page. This function assumes that thrashing has been just reported on
// the page. It will fail otherwise.
uvm_processor_mask_t *uvm_perf_thrashing_get_thrashing_processors(uvm_va_block_t *va_block, NvU64 address);

const uvm_page_mask_t *uvm_perf_thrashing_get_thrashing_pages(uvm_va_block_t *va_block);

// Global initialization/cleanup functions
NV_STATUS uvm_perf_thrashing_init(void);
void uvm_perf_thrashing_exit(void);

// Per-GPU initialization/cleanup functions. See comments in
// uvm_perf_heuristics.h
NV_STATUS uvm_perf_thrashing_add_gpu(uvm_gpu_t *gpu);
void uvm_perf_thrashing_remove_gpu(uvm_gpu_t *gpu);

// VA space Initialization/cleanup functions. See comments in
// uvm_perf_heuristics.h
NV_STATUS uvm_perf_thrashing_load(uvm_va_space_t *va_space);
NV_STATUS uvm_perf_thrashing_register_gpu(uvm_va_space_t *va_space, uvm_gpu_t *gpu);
void uvm_perf_thrashing_stop(uvm_va_space_t *va_space);
void uvm_perf_thrashing_unload(uvm_va_space_t *va_space);

// Destroy the thrashing detection struct for the given block.
void uvm_perf_thrashing_info_destroy(uvm_va_block_t *va_block);

// Unmap remote mappings from all processors on the pinned pages described by
// region and block_thrashing->pinned pages.  va_block_context must not be NULL
// and policy for the region must match.  See the comments for
// uvm_va_block_check_policy_is_valid() in uvm_va_block.h.  Locking: the
// va_block lock must be held.
NV_STATUS uvm_perf_thrashing_unmap_remote_pinned_pages_all(uvm_va_block_t *va_block,
                                                           uvm_va_block_context_t *va_block_context,
                                                           uvm_va_block_region_t region);

// edit : JH - from uvm_perf_thrashing.c
// Number of bits for page-granularity time stamps. Currently we ignore the first 6 bits
// of the timestamp (i.e. we have 64ns resolution, which is good enough)
#define PAGE_THRASHING_LAST_TIME_STAMP_BITS 58
#define PAGE_THRASHING_NUM_EVENTS_BITS      3

#define PAGE_THRASHING_THROTTLING_END_TIME_STAMP_BITS 58
#define PAGE_THRASHING_THROTTLING_COUNT_BITS          8

// Per-page thrashing detection structure.
typedef struct
{
    struct
    {
        // Last time stamp when a thrashing-related event was recorded
        NvU64                        last_time_stamp : PAGE_THRASHING_LAST_TIME_STAMP_BITS;

        bool                    has_migration_events : 1;

        bool                   has_revocation_events : 1;

        // Number of consecutive "thrashing" events (within the configured
        // thrashing lapse)
        NvU8                    num_thrashing_events : PAGE_THRASHING_NUM_EVENTS_BITS;

        bool                                  pinned : 1;
    };

    struct
    {
        // Deadline for throttled processors to wake up
        NvU64              throttling_end_time_stamp : PAGE_THRASHING_THROTTLING_END_TIME_STAMP_BITS;

        // Number of times a processor has been throttled. This is used to
        // determine when the page needs to get pinned. After getting pinned
        // this field is always 0.
        NvU8                        throttling_count : PAGE_THRASHING_THROTTLING_COUNT_BITS;
    };

    // Processors accessing this page
    uvm_processor_mask_t                  processors;

    // Processors that have been throttled. This must be a subset of processors
    uvm_processor_mask_t        throttled_processors;

    // Memory residency for the page when in pinning phase
    uvm_processor_id_t           pinned_residency_id;

    // Processor not to be throttled in the current throttling period
    uvm_processor_id_t  do_not_throttle_processor_id;
} page_thrashing_info_t;

// Per-VA block thrashing detection structure. This state is protected by the
// VA block lock.
typedef struct
{
    page_thrashing_info_t                     *pages;

    NvU16                        num_thrashing_pages;

    NvU8                       thrashing_reset_count;

    uvm_processor_id_t                last_processor;

    NvU64                            last_time_stamp;

    NvU64                  last_thrashing_time_stamp;

    // Stats
    NvU32                           throttling_count;

    uvm_page_mask_t                  thrashing_pages;

    struct
    {
        NvU32                                  count;

        uvm_page_mask_t                         mask;

        // List of pinned pages. This list is only used if the pinning timeout
        // is not 0.
        struct list_head                        list;
    } pinned_pages;
} block_thrashing_info_t;

block_thrashing_info_t *thrashing_info_get(uvm_va_block_t *va_block); // edit : JH
//--

#endif