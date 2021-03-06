/*******************************************************************************
    Copyright (c) 2015-2019 NVIDIA Corporation

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

#ifndef __UVM8_GLOBAL_H__
#define __UVM8_GLOBAL_H__

#include "nv_uvm_types.h"
#include "uvm8_extern_decl.h"
#include "uvm_linux.h"
#include "uvm_common.h"
#include "uvm8_processors.h"
#include "uvm8_gpu.h"
#include "uvm8_lock.h"
#include "uvm8_ats_ibm.h"

// Global state of the uvm driver
struct uvm_global_struct
{
    // Mask of retained GPUs.
    // Note that GPUs are added to this mask as the last step of add_gpu() and
    // removed from it as the first step of remove_gpu() implying that a GPU
    // that's being initialized or deinitialized will not be in it.
    uvm_global_processor_mask_t retained_gpus;

    // Array of the GPUs retained by uvm
    // Note that GPUs will have ids offset by 1 to accomodate the UVM_ID_CPU
    // so e.g. gpus[0] will have GPU id = 1.
    // A GPU entry is unused iff it does not exist (is a NULL pointer) in this table.
    uvm_gpu_t *gpus[UVM_GLOBAL_ID_MAX_GPUS];

    // A global RM session (RM client)
    // Created on module load and destroyed on module unload
    uvmGpuSessionHandle rm_session_handle;

    // peer-to-peer table
    // peer info is added and removed from this table when usermode
    // driver calls UvmEnablePeerAccess and UvmDisablePeerAccess
    // respectively.
    uvm_gpu_peer_t peers[UVM_MAX_UNIQUE_GPU_PAIRS];

    // Stores an NV_STATUS, once it becomes != NV_OK, the driver should refuse to
    // do most anything other than try and clean up as much as possible.
    // An example of a fatal error is an unrecoverable ECC error on one of the
    // GPUs.
    atomic_t fatal_error;

    // A flag to disable the assert on fatal error
    // To be used by tests and only consulted if tests are enabled.
    bool disable_fatal_error_assert;

    // Lock protecting the global state
    uvm_mutex_t global_lock;

    struct
    {
        // Lock synchronizing user threads with power management activity
        uvm_rw_semaphore_t lock;

        // Power management state flag; tested by UVM_GPU_WRITE_ONCE()
        // and UVM_GPU_READ_ONCE() to detect accesses to GPUs when
        // UVM is suspended.
        bool is_suspended;
    } pm;

    // This lock synchronizes addition and removal of GPUs from UVM's global table.
    // It must be held whenever g_uvm_global.gpus[] is written. In order to read
    // from this table, you must hold either the gpu_table_lock, or the global_lock.
    //
    // This is a leaf lock.
    uvm_spinlock_irqsave_t gpu_table_lock;

    // Number of simulated/emulated devices that have registered with UVM
    unsigned num_simulated_devices;

    // A single queue for deferred work that is non-GPU-specific.
    nv_kthread_q_t global_q;

    // A single queue for deferred f_ops->release() handling.  Items scheduled to
    // run on it may block for the duration of system sleep cycles, stalling
    // the queue and preventing any other items from running.
    nv_kthread_q_t deferred_release_q;

    struct
    {
        // Indicates whether the system HW supports ATS. This field is set once
        // during global initialization (uvm_global_init), and can be read
        // afterwards without acquiring any locks.
        bool supported;

        // On top of HW platform support, ATS support can be overridden using
        // the module parameter uvm8_ats_mode. This field is set once during
        // global initialization (uvm_global_init), and can be read afterwards
        // without acquiring any locks.
        bool enabled;
    } ats;

#if UVM_IBM_NPU_SUPPORTED()
    // On IBM systems this array tracks the active NPUs (the NPUs which are
    // attached to retained GPUs).
    uvm_ibm_npu_t npus[NV_MAX_NPUS];
#endif

    // List of all active VA spaces
    struct
    {
        uvm_mutex_t lock;
        struct list_head list;
    } va_spaces;
};

// Initialize global uvm state
NV_STATUS uvm_global_init(void);

// Deinitialize global state (called from module exit)
void uvm_global_exit(void);

// Prepare for entry into a system sleep state
NV_STATUS uvm_suspend_entry(void);

// Recover after exit from a system sleep state
NV_STATUS uvm_resume_entry(void);

// Get a gpu by its global id.
// Returns a pointer to the GPU object, or NULL if not found.
//
// LOCKING: requires that you hold the gpu_table_lock, the global_lock, or have
// retained the gpu.
static uvm_gpu_t *uvm_gpu_get(uvm_global_gpu_id_t gpu_id)
{
    return g_uvm_global.gpus[uvm_global_id_gpu_index(gpu_id)];
}

// Get a gpu by its processor id.
// Returns a pointer to the GPU object, or NULL if not found.
//
// LOCKING: requires that you hold the gpu_table_lock, the global_lock, or have
// retained the gpu.







static uvm_gpu_t *uvm_gpu_get_by_processor_id(uvm_processor_id_t id)
{
    uvm_global_gpu_id_t global_id = uvm_global_gpu_id(uvm_id_value(id));
    uvm_gpu_t *gpu = uvm_gpu_get(global_id);

    UVM_ASSERT(uvm_id_value(gpu->id) == uvm_global_id_value(gpu->global_id));

    return gpu;
}

static uvmGpuSessionHandle uvm_gpu_session_handle(uvm_gpu_t *gpu)
{




    return g_uvm_global.rm_session_handle;
}

// Use these READ_ONCE()/WRITE_ONCE() wrappers when accessing GPU resources
// in BAR0/BAR1 to detect cases in which GPUs are accessed when UVM is
// suspended.
#define UVM_GPU_WRITE_ONCE(x, val) do {         \
        UVM_ASSERT(!uvm_global_is_suspended()); \
        UVM_WRITE_ONCE(x, val);                 \
    } while (0)

#define UVM_GPU_READ_ONCE(x) ({                 \
        UVM_ASSERT(!uvm_global_is_suspended()); \
        UVM_READ_ONCE(x);                       \
    })

static bool global_is_fatal_error_assert_disabled(void)
{
    // Only allow the assert to be disabled if tests are enabled
    if (!uvm_enable_builtin_tests)
        return false;

    return g_uvm_global.disable_fatal_error_assert;
}

// Set a global fatal error
// Once that happens the the driver should refuse to do anything other than try
// and clean up as much as possible.
// An example of a fatal error is an unrecoverable ECC error on one of the
// GPUs.
// Use a macro so that the assert below provides precise file and line info and
// a backtrace.
#define uvm_global_set_fatal_error(error)                                       \
    do {                                                                        \
        if (!global_is_fatal_error_assert_disabled())                           \
            UVM_ASSERT_MSG(0, "Fatal error: %s\n", nvstatusToString(error));    \
        uvm_global_set_fatal_error_impl(error);                                 \
    } while (0)
void uvm_global_set_fatal_error_impl(NV_STATUS error);

// Get the global status
static NV_STATUS uvm_global_get_status(void)
{
    return atomic_read(&g_uvm_global.fatal_error);
}

// Reset global fatal error
// This is to be used by tests triggering the global error on purpose only.
// Returns the value of the global error field that existed just before this
// reset call was made.
NV_STATUS uvm_global_reset_fatal_error(void);

static uvm_gpu_t *uvm_global_processor_mask_find_first_gpu(const uvm_global_processor_mask_t *global_gpus)
{
    uvm_gpu_t *gpu;
    uvm_global_gpu_id_t gpu_id = uvm_global_processor_mask_find_first_gpu_id(global_gpus);

    if (UVM_GLOBAL_ID_IS_INVALID(gpu_id))
        return NULL;

    gpu = uvm_gpu_get(gpu_id);

    // If there is valid GPU id in the mask, assert that the corresponding
    // uvm_gpu_t is present. Otherwise it would stop a
    // for_each_global_gpu_in_mask() loop pre-maturely. Today, this could only
    // happen in remove_gpu() because the GPU being removed is deleted from the
    // global table very early.
    UVM_ASSERT_MSG(gpu, "gpu_id %u\n", uvm_global_id_value(gpu_id));

    return gpu;
}

static uvm_gpu_t *__uvm_global_processor_mask_find_next_gpu(const uvm_global_processor_mask_t *global_gpus, uvm_gpu_t *gpu)
{
    uvm_global_gpu_id_t gpu_id;

    UVM_ASSERT(gpu);

    gpu_id = uvm_global_processor_mask_find_next_id(global_gpus, uvm_global_gpu_id_next(gpu->global_id));
    if (UVM_GLOBAL_ID_IS_INVALID(gpu_id))
        return NULL;

    gpu = uvm_gpu_get(gpu_id);

    // See comment in uvm_global_processor_mask_find_first_gpu().
    UVM_ASSERT_MSG(gpu, "gpu_id %u\n", uvm_global_id_value(gpu_id));

    return gpu;
}

static uvm_gpu_t *uvm_global_processor_mask_find_next_gpu(const uvm_global_processor_mask_t *global_gpus, uvm_gpu_t *gpu)
{
    if (gpu == NULL)
        return NULL;

    return __uvm_global_processor_mask_find_next_gpu(global_gpus, gpu);
}

// Helper to iterate over all GPUs in the input mask
#define for_each_global_gpu_in_mask(gpu, global_mask)                                         \
    for (gpu = uvm_global_processor_mask_find_first_gpu(global_mask);                         \
         gpu != NULL;                                                                         \
         gpu = __uvm_global_processor_mask_find_next_gpu(global_mask, gpu))

// Helper to iterate over all GPUs retained by the UVM driver (across all va spaces)
#define for_each_global_gpu(gpu)                                                              \
    for (({uvm_assert_mutex_locked(&g_uvm_global.global_lock);                                \
           gpu = uvm_global_processor_mask_find_first_gpu(&g_uvm_global.retained_gpus);});    \
           gpu != NULL;                                                                       \
           gpu = __uvm_global_processor_mask_find_next_gpu(&g_uvm_global.retained_gpus, gpu))

// Helper which calls uvm_gpu_retain on each GPU in mask
void uvm_global_mask_retain(const uvm_global_processor_mask_t *mask);

// Helper which calls uvm_gpu_release_locked on each GPU in mask.
//
// LOCKING: this function takes and releases the global lock if the input mask
//          is not empty
void uvm_global_mask_release(const uvm_global_processor_mask_t *mask);

// Check for ECC errors for all GPUs in a mask
// Notably this check cannot be performed where it's not safe to call into RM.
NV_STATUS uvm_global_mask_check_ecc_error(uvm_global_processor_mask_t *gpus);

#endif // __UVM8_GLOBAL_H__
