/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#include "nvidia-drm-priv.h"
#include "nvidia-drm-ioctl.h"
#include "nvidia-drm-prime-fence.h"
#include "nvidia-drm-gem.h"
#include "nvidia-drm-gem-nvkms-memory.h"
#include "nvidia-drm-gem-user-memory.h"
#include "nvidia-dma-resv-helper.h"

#if defined(NV_DRM_DRM_DRV_H_PRESENT)
#include <drm/drm_drv.h>
#endif

#if defined(NV_DRM_DRM_PRIME_H_PRESENT)
#include <drm/drm_prime.h>
#endif

#include "linux/dma-buf.h"

#include "nv-mm.h"

void nv_drm_gem_free(struct drm_gem_object *gem)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    /* Cleanup core gem object */
    drm_gem_object_release(&nv_gem->base);

#if defined(NV_DRM_FENCE_AVAILABLE) && !defined(NV_DRM_GEM_OBJECT_HAS_RESV)
    nv_dma_resv_fini(&nv_gem->resv);
#endif

    nv_gem->ops->free(nv_gem);
}

#if !defined(NV_DRM_DRIVER_HAS_GEM_FREE_OBJECT)
static struct drm_gem_object_funcs nv_drm_gem_funcs = {
    .free = nv_drm_gem_free,
    .get_sg_table = nv_drm_gem_prime_get_sg_table,
};
#endif

void nv_drm_gem_object_init(struct nv_drm_device *nv_dev,
                            struct nv_drm_gem_object *nv_gem,
                            const struct nv_drm_gem_object_funcs * const ops,
                            size_t size)
{
    struct drm_device *dev = nv_dev->dev;

    nv_gem->nv_dev = nv_dev;
    nv_gem->ops = ops;

    /* Initialize the gem object */

#if defined(NV_DRM_FENCE_AVAILABLE)
    nv_dma_resv_init(&nv_gem->resv);

#if defined(NV_DRM_GEM_OBJECT_HAS_RESV)
    nv_gem->base.resv = &nv_gem->resv;
#endif

#endif

#if !defined(NV_DRM_DRIVER_HAS_GEM_FREE_OBJECT)
    nv_gem->base.funcs = &nv_drm_gem_funcs;
#endif

    drm_gem_private_object_init(dev, &nv_gem->base, size);
}

struct drm_gem_object *nv_drm_gem_prime_import(struct drm_device *dev,
                                               struct dma_buf *dma_buf)
{
#if defined(NV_DMA_BUF_OWNER_PRESENT)
    struct drm_gem_object *gem_dst;
    struct nv_drm_gem_object *nv_gem_src;

    if (dma_buf->owner == dev->driver->fops->owner) {
        nv_gem_src = to_nv_gem_object(dma_buf->priv);

        if (nv_gem_src->base.dev != dev &&
            nv_gem_src->ops->prime_dup != NULL) {
            /*
             * If we're importing from another NV device, try to handle the
             * import internally rather than attaching through the dma-buf
             * mechanisms.  Importing from the same device is even easier,
             * and drm_gem_prime_import() handles that just fine.
             */
            gem_dst = nv_gem_src->ops->prime_dup(dev, nv_gem_src);

            if (gem_dst)
                return gem_dst;
        }
    }
#endif /* NV_DMA_BUF_OWNER_PRESENT */

    return drm_gem_prime_import(dev, dma_buf);
}

struct drm_gem_object *nv_drm_gem_prime_import_sg_table(
    struct drm_device *dev,
    struct dma_buf_attachment *attach,
    struct sg_table *sgt)
{
    struct nv_drm_gem_user_memory *nv_user_memory;

    nv_user_memory =
        nv_drm_gem_user_memory_import_sg_table(dev, attach->dmabuf, sgt);
    if (!nv_user_memory) {
        return NULL;
    }

    return &nv_user_memory->base.base;
}

struct sg_table *nv_drm_gem_prime_get_sg_table(struct drm_gem_object *gem)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (nv_gem->ops->prime_get_sg_table != NULL) {
        return nv_gem->ops->prime_get_sg_table(nv_gem);
    }

    return ERR_PTR(-ENOTSUPP);
}

void *nv_drm_gem_prime_vmap(struct drm_gem_object *gem)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (nv_gem->ops->prime_vmap != NULL) {
        return nv_gem->ops->prime_vmap(nv_gem);
    }

    return ERR_PTR(-ENOTSUPP);
}

void nv_drm_gem_prime_vunmap(struct drm_gem_object *gem, void *address)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (nv_gem->ops->prime_vunmap != NULL) {
        nv_gem->ops->prime_vunmap(nv_gem, address);
    }
}

#if defined(NV_DRM_DRIVER_HAS_GEM_PRIME_RES_OBJ)
nv_dma_resv_t* nv_drm_gem_prime_res_obj(struct drm_gem_object *obj)
{
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(obj);

    return &nv_gem->resv;
}
#endif

int nv_drm_gem_map_offset_ioctl(struct drm_device *dev,
                                void *data, struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_gem_map_offset_params *params = data;
    struct nv_drm_gem_object *nv_gem;
    int ret;

    if ((nv_gem = nv_drm_gem_object_lookup(dev,
                                           filep,
                                           params->handle)) == NULL) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to lookup gem object for map: 0x%08x",
            params->handle);
        return -EINVAL;
    }

    if (nv_gem->ops->create_mmap_offset) {
        ret = nv_gem->ops->create_mmap_offset(nv_dev, nv_gem, &params->offset);
    } else {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Gem object type does not support mapping: 0x%08x",
            params->handle);
        ret = -EINVAL;
    }

    nv_drm_gem_object_unreference_unlocked(nv_gem);

    return ret;
}

/* XXX Move these vma operations to os layer */

static vm_fault_t __nv_drm_vma_fault(struct vm_area_struct *vma,
                              struct vm_fault *vmf)
{
    struct drm_gem_object *gem = vma->vm_private_data;
    struct nv_drm_gem_object *nv_gem = to_nv_gem_object(gem);

    if (!nv_gem) {
        return VM_FAULT_SIGBUS;
    }

    return nv_gem->ops->handle_vma_fault(nv_gem, vma, vmf);
}

/*
 * Note that nv_drm_vma_fault() can be called for different or same
 * ranges of the same drm_gem_object simultaneously.
 */

#if defined(NV_VM_OPS_FAULT_REMOVED_VMA_ARG)
static vm_fault_t nv_drm_vma_fault(struct vm_fault *vmf)
{
    return __nv_drm_vma_fault(vmf->vma, vmf);
}
#else
static vm_fault_t nv_drm_vma_fault(struct vm_area_struct *vma,
                                struct vm_fault *vmf)
{
    return __nv_drm_vma_fault(vma, vmf);
}
#endif

const struct vm_operations_struct nv_drm_gem_vma_ops = {
    .open  = drm_gem_vm_open,
    .fault = nv_drm_vma_fault,
    .close = drm_gem_vm_close,
};

#endif /* NV_DRM_AVAILABLE */
