/*******************************************************************************
    Copyright (c) 2017-2019 NVIDIA Corporation

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

#include "uvm8_hal.h"
#include "uvm8_gpu.h"
#include "uvm8_mem.h"
#include "uvm8_turing_fault_buffer.h"

void uvm_hal_turing_arch_init_properties(uvm_gpu_t *gpu)
{
    gpu->big_page.swizzling = false;

    gpu->tlb_batch.va_invalidate_supported = true;

    gpu->tlb_batch.va_range_invalidate_supported = true;

    // TODO: Bug 1767241: Run benchmarks to figure out a good number
    gpu->tlb_batch.max_ranges = 8;

    gpu->utlb_per_gpc_count = uvm_turing_get_utlbs_per_gpc(gpu);

    gpu->fault_buffer_info.replayable.utlb_count = gpu->rm_info.gpcCount * gpu->utlb_per_gpc_count;
    {
        uvm_fault_buffer_entry_t *dummy;
        UVM_ASSERT(gpu->fault_buffer_info.replayable.utlb_count <= (1 << (sizeof(dummy->fault_source.utlb_id) * 8)));
    }

    // A single top level PDE on Turing covers 128 TB and that's the minimum
    // size that can be used.
    gpu->rm_va_base = 0;
    gpu->rm_va_size = 128ull * 1024 * 1024 * 1024 * 1024;

    gpu->uvm_mem_va_base = 384ull * 1024 * 1024 * 1024 * 1024;
    gpu->uvm_mem_va_size = UVM_MEM_VA_SIZE;

    gpu->peer_copy_mode = UVM_GPU_PEER_COPY_MODE_VIRTUAL;

    // Not all units on Turing support 49-bit addressing, including those which
    // access channel buffers.
    gpu->max_channel_va = 1ULL << 40;

    // Turing can map sysmem with any page size
    gpu->can_map_sysmem_with_large_pages = true;

    // Prefetch instructions will generate faults
    gpu->prefetch_fault_supported = true;

    // Turing can place GPFIFO in vidmem
    gpu->gpfifo_in_vidmem_supported = true;

    gpu->replayable_faults_supported = true;

    gpu->non_replayable_faults_supported = true;

    gpu->access_counters_supported = true;

    gpu->fault_cancel_va_supported = true;

    gpu->scoped_atomics_supported = true;

    gpu->has_pulse_based_interrupts = true;

    gpu->has_clear_faulted_channel_method = true;

    gpu->sparse_mappings_supported = true;




}
