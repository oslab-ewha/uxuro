/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */
 
#include "nv-msi.h"
#include "nv-proto.h"

#if defined(NV_LINUX_PCIE_MSI_SUPPORTED)
void NV_API_CALL nv_init_msi(nv_state_t *nv)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    int rc = 0;

    rc = pci_enable_msi(nvl->pci_dev);
    if (rc == 0)
    {
        nv->interrupt_line = nvl->pci_dev->irq;
        nv->flags |= NV_FLAG_USES_MSI;
        nvl->num_intr = 1;
    }
    else
    {
        nv->flags &= ~NV_FLAG_USES_MSI;
        if (nvl->pci_dev->irq != 0)
        {
            NV_DEV_PRINTF(NV_DBG_ERRORS, nv,
                      "Failed to enable MSI; "
                      "falling back to PCIe virtual-wire interrupts.\n");
        }
    }

    return;
}

void NV_API_CALL nv_init_msix(nv_state_t *nv)
{
    nv_linux_state_t *nvl = NV_GET_NVL_FROM_NV_STATE(nv);
    int num_intr = 0;
    struct msix_entry *msix_entries;
    int rc = 0;
    int i;

    NV_SPIN_LOCK_INIT(&nvl->msix_isr_lock);

    rc = os_alloc_mutex(&nvl->msix_bh_mutex);
    if (rc != 0)
        goto failed;

    num_intr = nv_get_max_irq(nvl->pci_dev);

    if (num_intr > NV_RM_MAX_MSIX_LINES)
    {
        NV_DEV_PRINTF(NV_DBG_INFO, nv, "Reducing MSI-X count from %d to the "
                               "driver-supported maximum %d.\n", num_intr, NV_RM_MAX_MSIX_LINES);
        num_intr = NV_RM_MAX_MSIX_LINES;
    }

    NV_KMALLOC(nvl->msix_entries, sizeof(struct msix_entry) * num_intr);
    if (nvl->msix_entries == NULL)
    {
        NV_DEV_PRINTF(NV_DBG_ERRORS, nv, "Failed to allocate MSI-X entries.\n");
        return;
    }

    for (i = 0, msix_entries = nvl->msix_entries; i < num_intr; i++, msix_entries++)
    {
        msix_entries->entry = i;
    }

    rc = nv_pci_enable_msix(nvl, num_intr);
    if (rc != NV_OK)
        goto failed;

    nv->flags |= NV_FLAG_USES_MSIX;
    return;

failed:
    nv->flags &= ~NV_FLAG_USES_MSIX;

    if (nvl->msix_entries)
    {
        NV_KFREE(nvl->msix_entries, sizeof(struct msix_entry) * num_intr);
    }

    if (nvl->msix_bh_mutex)
    {
        os_free_mutex(nvl->msix_bh_mutex);
        nvl->msix_bh_mutex = NULL;
    }
    NV_DEV_PRINTF(NV_DBG_ERRORS, nv, "Failed to enable MSI-X.\n");
}

NvS32 NV_API_CALL nv_request_msix_irq(nv_linux_state_t *nvl)
{
    int i;
    int j;
    struct msix_entry *msix_entries;
    int rc = NV_ERR_INVALID_ARGUMENT;
    nv_state_t *nv = NV_STATE_PTR(nvl);

    for (i = 0, msix_entries = nvl->msix_entries; i < nvl->num_intr;
         i++, msix_entries++)
    {
        rc = request_threaded_irq(msix_entries->vector, nvidia_isr_msix,
                                  nvidia_isr_msix_kthread_bh, nv_default_irq_flags(nv),
                                  nv_device_name, (void *)nvl);
        if (rc)
        {
            for( j = 0; j < i; j++)
            {
                free_irq(nvl->msix_entries[i].vector, (void *)nvl);
            }
            break;
        }
    }

    return rc;
}
#endif