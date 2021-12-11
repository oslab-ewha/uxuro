/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#if defined(NV_DRM_DRM_PRIME_H_PRESENT)
#include <drm/drm_prime.h>
#endif

#include "nvidia-drm-gem-user-memory.h"
#include "nvidia-drm-ioctl.h"

#include "linux/dma-buf.h"
#include "linux/mm.h"
#include "nv-mm.h"

static inline
void __nv_drm_gem_user_memory_free(struct nv_drm_gem_object *nv_gem)
{
    struct nv_drm_gem_user_memory *nv_user_memory = to_nv_user_memory(nv_gem);

    if (nv_gem->base.import_attach) {
        BUG_ON(!nv_user_memory->sgt);

        drm_prime_gem_destroy(&nv_gem->base, nv_user_memory->sgt);
        nv_drm_free(nv_user_memory->pages);
    } else {
        BUG_ON(nv_user_memory->sgt);

        nv_drm_unlock_user_pages(nv_user_memory->pages_count,
                                 nv_user_memory->pages);
    }

    nv_drm_free(nv_user_memory);
}

static struct sg_table *__nv_drm_gem_user_memory_prime_get_sg_table(
    struct nv_drm_gem_object *nv_gem)
{
    struct nv_drm_gem_user_memory *nv_user_memory = to_nv_user_memory(nv_gem);

    return drm_prime_pages_to_sg(nv_user_memory->pages,
                                 nv_user_memory->pages_count);
}

static void *__nv_drm_gem_user_memory_prime_vmap(
    struct nv_drm_gem_object *nv_gem)
{
    struct nv_drm_gem_user_memory *nv_user_memory = to_nv_user_memory(nv_gem);

    return nv_drm_vmap(nv_user_memory->pages,
                           nv_user_memory->pages_count);
}

static void __nv_drm_gem_user_memory_prime_vunmap(
    struct nv_drm_gem_object *gem,
    void *address)
{
    nv_drm_vunmap(address);
}

static bool __nv_drm_gem_user_memory_adjust_mmap_flags(
    struct nv_drm_gem_object *nv_gem,
    struct vm_area_struct *vma)
{
    /*
     * Enforce that user-memory GEM mappings are MAP_SHARED, to prevent COW
     * with MAP_PRIVATE and VM_MIXEDMAP
     */
    if (!(vma->vm_flags & VM_SHARED)) {
        return false;
    }

    vma->vm_flags &= ~VM_PFNMAP;
    vma->vm_flags &= ~VM_IO;
    vma->vm_flags |= VM_MIXEDMAP;

    return true;
}

static vm_fault_t __nv_drm_gem_user_memory_handle_vma_fault(
    struct nv_drm_gem_object *nv_gem,
    struct vm_area_struct *vma,
    struct vm_fault *vmf)
{
    struct nv_drm_gem_user_memory *nv_user_memory = to_nv_user_memory(nv_gem);
    unsigned long address = nv_page_fault_va(vmf);
    struct drm_gem_object *gem = vma->vm_private_data;
    unsigned long page_offset;
    vm_fault_t ret;

    page_offset = vmf->pgoff - drm_vma_node_start(&gem->vma_node);

    BUG_ON(page_offset > nv_user_memory->pages_count);

    ret = vm_insert_page(vma, address, nv_user_memory->pages[page_offset]);
    switch (ret) {
        case 0:
        case -EBUSY:
            /*
             * EBUSY indicates that another thread already handled
             * the faulted range.
             */
            ret = VM_FAULT_NOPAGE;
            break;
        case -ENOMEM:
            ret = VM_FAULT_OOM;
            break;
        default:
            WARN_ONCE(1, "Unhandled error in %s: %d\n", __FUNCTION__, ret);
            ret = VM_FAULT_SIGBUS;
            break;
    }

    return ret;
}

static int __nv_drm_gem_user_create_mmap_offset(
    struct nv_drm_device *nv_dev,
    struct nv_drm_gem_object *nv_gem,
    uint64_t *offset)
{
    (void)nv_dev;
    return nv_drm_gem_create_mmap_offset(nv_gem, offset);
}

const struct nv_drm_gem_object_funcs __nv_gem_user_memory_ops = {
    .free = __nv_drm_gem_user_memory_free,
    .prime_get_sg_table = __nv_drm_gem_user_memory_prime_get_sg_table,
    .prime_vmap = __nv_drm_gem_user_memory_prime_vmap,
    .prime_vunmap = __nv_drm_gem_user_memory_prime_vunmap,
    .adjust_mmap_flags = __nv_drm_gem_user_memory_adjust_mmap_flags,
    .handle_vma_fault = __nv_drm_gem_user_memory_handle_vma_fault,
    .create_mmap_offset = __nv_drm_gem_user_create_mmap_offset,
};

struct nv_drm_gem_user_memory *nv_drm_gem_user_memory_import_sg_table(
    struct drm_device *dev,
    struct dma_buf *dma_buf,
    struct sg_table *sgt)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct nv_drm_gem_user_memory *nv_user_memory;

    struct page **pages = NULL;
    unsigned long pages_count = 0;

    if ((nv_user_memory =
            nv_drm_calloc(1, sizeof(*nv_user_memory))) == NULL) {
        return NULL;
    }

    // dma_buf->size must be a multiple of PAGE_SIZE
    BUG_ON(dma_buf->size % PAGE_SIZE);

    pages_count = dma_buf->size >> PAGE_SHIFT;
    if ((pages =
            nv_drm_calloc(pages_count, sizeof(*pages))) == NULL) {
        nv_drm_free(nv_user_memory);
        return NULL;
    }

    if (drm_prime_sg_to_page_addr_arrays(sgt, pages, NULL, pages_count) < 0) {
        nv_drm_free(nv_user_memory);
        nv_drm_free(pages);
        return NULL;
    }

    nv_user_memory->pages = pages;
    nv_user_memory->pages_count = pages_count;

    nv_user_memory->sgt = sgt;

    nv_drm_gem_object_init(nv_dev,
                           &nv_user_memory->base,
                           &__nv_gem_user_memory_ops,
                           dma_buf->size);

    return nv_user_memory;
}

int nv_drm_gem_import_userspace_memory_ioctl(struct drm_device *dev,
                                             void *data, struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);

    struct drm_nvidia_gem_import_userspace_memory_params *params = data;
    struct nv_drm_gem_user_memory *nv_user_memory;

    struct page **pages = NULL;
    unsigned long pages_count = 0;

    int ret = 0;

    if ((params->size % PAGE_SIZE) != 0) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Userspace memory 0x%llx size should be in a multiple of page "
            "size to create a gem object",
            params->address);
        return -EINVAL;
    }

    pages_count = params->size / PAGE_SIZE;

    ret = nv_drm_lock_user_pages(params->address, pages_count, &pages);

    if (ret != 0) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to lock user pages for address 0x%llx: %d",
            params->address, ret);
        return ret;
    }

    if ((nv_user_memory =
            nv_drm_calloc(1, sizeof(*nv_user_memory))) == NULL) {
        ret = -ENOMEM;
        goto failed;
    }

    nv_user_memory->pages = pages;
    nv_user_memory->pages_count = pages_count;

    nv_drm_gem_object_init(nv_dev,
                           &nv_user_memory->base,
                           &__nv_gem_user_memory_ops,
                           params->size);

    return nv_drm_gem_handle_create_drop_reference(filep,
                                                   &nv_user_memory->base,
                                                   &params->handle);

failed:
    nv_drm_unlock_user_pages(pages_count, pages);

    return ret;
}

#endif
