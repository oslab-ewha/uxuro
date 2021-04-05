/*******************************************************************************
    Copyright (c) 2016-2019 NVIDIA Corporation

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

#include <linux/version.h>

#include "conftest.h"
#include "nvmisc.h"
#include "nv-linux.h"
#include "nv-procfs.h"
#include "nvlink_common.h"
#include "nvlink_errors.h"
#include "nv-kthread-q.h"

#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/poll.h>
#include <linux/sched.h>
#include <linux/time.h>
#include <linux/string.h>
#include <linux/moduleparam.h>
#include <linux/ctype.h>
#include <linux/wait.h>
#include <linux/jiffies.h>

#include "export_nvswitch.h"
#include "ioctl_nvswitch.h"

const static struct
{
    NvlStatus status;
    int err;
} nvswitch_status_map[] = {
    { NVL_ERR_GENERIC,          -EIO        },
    { NVL_NO_MEM,               -ENOMEM     },
    { NVL_BAD_ARGS ,            -EINVAL     },
    { NVL_ERR_INVALID_STATE,    -EIO        },
    { NVL_ERR_NOT_SUPPORTED,    -EOPNOTSUPP },
    { NVL_NOT_FOUND ,           -EINVAL     },
    { NVL_ERR_STATE_IN_USE,     -EBUSY      },
    { NVL_ERR_NOT_IMPLEMENTED,  -ENOSYS     },
    { NVL_SUCCESS,               0          },
};

static int
nvswitch_map_status
(
    NvlStatus status
)
{
    int err = -EIO;
    NvU32 i;
    NvU32 limit = sizeof(nvswitch_status_map) / sizeof(nvswitch_status_map[0]);

    for (i = 0; i < limit; i++)
    {
        if (nvswitch_status_map[i].status == status ||
            nvswitch_status_map[i].status == -status)
        {
            err = nvswitch_status_map[i].err;
            break;
        }
    }

    return err;
}

#if !defined(IRQF_SHARED)
#define IRQF_SHARED SA_SHIRQ
#endif

#if defined(NV_FILE_HAS_INODE)
#define NV_FILE_INODE(file) (file)->f_inode
#else
#define NV_FILE_INODE(file) (file)->f_dentry->d_inode
#endif

static int nvswitch_probe(struct pci_dev *, const struct pci_device_id *);
static void nvswitch_remove(struct pci_dev *);

static struct pci_device_id nvswitch_pci_table[] =
{
    {
        .vendor      = PCI_VENDOR_ID_NVIDIA,
        .device      = PCI_ANY_ID,
        .subvendor   = PCI_ANY_ID,
        .subdevice   = PCI_ANY_ID,
        .class       = (PCI_CLASS_BRIDGE_OTHER << 8),
        .class_mask  = ~0
    },
    {}
};

static struct pci_driver nvswitch_pci_driver =
{
    .name           = NVSWITCH_DRIVER_NAME,
    .id_table       = nvswitch_pci_table,
    .probe          = nvswitch_probe,
    .remove         = nvswitch_remove,
    .shutdown       = nvswitch_remove
};

//
// nvidia_nvswitch_mknod uses minor number 255 to create nvidia-nvswitchctl
// node. Hence, if NVSWITCH_CTL_MINOR is changed, then NV_NVSWITCH_CTL_MINOR
// should be updated. See nvdia-modprobe-utils.h
//
#define NVSWITCH_CTL_MINOR 255
#define NVSWITCH_MINOR_COUNT (NVSWITCH_CTL_MINOR + 1)

#define NVSWITCH_SHORT_NAME "nvswi"

// 32 bit hex value - including 0x prefix. (10 chars)
#define NVSWITCH_REGKEY_VALUE_LEN 10

static char *NvSwitchRegDwords;
module_param(NvSwitchRegDwords, charp, 0);
MODULE_PARM_DESC(NvSwitchRegDwords, "NvSwitch regkey");

//
// Locking:
//   We handle nvswitch driver locking in the OS layer. The nvswitch lib
//   layer does not have its own locking. It relies on the OS layer for
//   atomicity.
//
//   All locking is done with sleep locks. We use threaded MSI interrupts to
//   facilitate this.
//
//   When handling a request from a user context we use the interruptible
//   version to enable a quick ^C return if there is lock contention.
//
//   nvswitch.driver_mutex is used to protect driver's global state, "struct
//   NVSWITCH". The driver_mutex is taken during .probe, .remove, .open,
//   .close, and nvswitch-ctl .ioctl operations.
//
//   nvswitch_dev.device_mutex is used to protect per-device state, "struct
//   NVSWITCH_DEV", once a device is opened. The device_mutex is taken during
//   .ioctl, .poll and other background tasks.
//
//   The kernel guarantees that .close won't happen while .ioctl and .poll
//   are going on and without successful .open one can't execute any file ops.
//   This behavior guarantees correctness of the locking model.
//
//   If .close is invoked and holding the lock which is also used by threaded
//   tasks such as interrupt, driver will deadlock while trying to stop such
//   tasks. For example, when threaded interrupts are enabled, free_irq() calls
//   kthread_stop() to flush pending interrupt tasks. The locking model
//   makes sure that such deadlock cases don't happen.
//
// Lock ordering:
//   nvswitch.driver_mutex
//   nvswitch_dev.device_mutex
//

// Per-chip driver state
typedef struct
{
    char name[sizeof(NVSWITCH_DRIVER_NAME) + 4];
    char sname[sizeof(NVSWITCH_SHORT_NAME) + 4];  /* short name */
    int minor;
    struct mutex device_mutex;
    nvswitch_device *lib_device;                  /* nvswitch library device */
    wait_queue_head_t wait_q_errors;
    NvBool msi;
    void *bar0;
    struct nv_kthread_q task_q;                   /* Background task queue */
    struct nv_kthread_q_item task_item;           /* Background dispatch task */
    atomic_t task_q_ready;
    wait_queue_head_t wait_q_shutdown;
    struct pci_dev *pci_dev;
    atomic_t ref_count;
    struct list_head list_node;
    NvBool unusable;
} NVSWITCH_DEV;

// Global driver state
typedef struct
{
    NvBool initialized;
    struct cdev cdev;
    struct cdev cdev_ctl;
    dev_t devno;
    atomic_t count;
    struct mutex driver_mutex;
    struct list_head devices;
} NVSWITCH;

static NVSWITCH nvswitch = {0};

static int nvswitch_device_open(struct inode *inode, struct file *file);
static int nvswitch_device_release(struct inode *inode, struct file *file);
static unsigned int nvswitch_device_poll(struct file *file, poll_table *wait);
static int nvswitch_device_ioctl(struct inode *inode,
                                 struct file *file,
                                 unsigned int cmd,
                                 unsigned long arg);
static long nvswitch_device_unlocked_ioctl(struct file *file,
                                           unsigned int cmd,
                                           unsigned long arg);

static int nvswitch_ctl_ioctl(struct inode *inode,
                              struct file *file,
                              unsigned int cmd,
                              unsigned long arg);
static long nvswitch_ctl_unlocked_ioctl(struct file *file,
                                        unsigned int cmd,
                                        unsigned long arg);

struct file_operations device_fops =
{
    .owner = THIS_MODULE,
#if defined(NV_FILE_OPERATIONS_HAS_IOCTL)
    .ioctl = nvswitch_device_ioctl,
#endif
    .unlocked_ioctl = nvswitch_device_unlocked_ioctl,
    .open    = nvswitch_device_open,
    .release = nvswitch_device_release,
    .poll    = nvswitch_device_poll
};

struct file_operations ctl_fops =
{
    .owner = THIS_MODULE,
#if defined(NV_FILE_OPERATIONS_HAS_IOCTL)
    .ioctl = nvswitch_ctl_ioctl,
#endif
    .unlocked_ioctl = nvswitch_ctl_unlocked_ioctl,
};

static int nvswitch_initialize_device_interrupt(NVSWITCH_DEV *nvswitch_dev);
static void nvswitch_shutdown_device_interrupt(NVSWITCH_DEV *nvswitch_dev);
static void nvswitch_load_bar_info(NVSWITCH_DEV *nvswitch_dev);
static void nvswitch_task_dispatch(NVSWITCH_DEV *nvswitch_dev);

#if defined(CONFIG_PROC_FS)
#define NV_DEFINE_SINGLE_NVSWITCH_PROCFS_FILE(name) \
    NV_DEFINE_SINGLE_PROCFS_FILE(name, \
                                 NV_READ_LOCK_SYSTEM_PM_LOCK_INTERRUPTIBLE, \
                                 NV_READ_UNLOCK_SYSTEM_PM_LOCK)
#endif

#define NVSWITCH_PROCFS_DIR "driver/nvidia-nvswitch"

static struct proc_dir_entry *nvswitch_procfs_dir = NULL;

#if defined(CONFIG_PROC_FS)
    static int nvswitch_is_procfs_available = 1;
#else
    static int nvswitch_is_procfs_available = 0;
#endif

static struct proc_dir_entry *nvswitch_permissions = NULL;

static int
nv_procfs_read_permissions
(
    struct seq_file *s,
    void *v
)
{
    // Restrict device node permissions to root-only - 0600.
    seq_printf(s, "%s: %u\n", "DeviceFileMode", 384);

    return 0;
}

NV_DEFINE_SINGLE_NVSWITCH_PROCFS_FILE(permissions);

static void
nvswitch_permissions_exit
(
    void
)
{
    if (!nvswitch_permissions)
    {
        return;
    }

    NV_REMOVE_PROC_ENTRY(nvswitch_permissions);
    nvswitch_permissions = NULL;
}

static int
nvswitch_permissions_init
(
    void
)
{
    if (!nvswitch_procfs_dir)
    {
        return -EACCES;
    }

    nvswitch_permissions = NV_CREATE_PROC_FILE("permissions",
                                               nvswitch_procfs_dir,
                                               permissions,
                                               NULL);
    if (!nvswitch_permissions)
    {
        return -EACCES;
    }

    return 0;
}

static void
nvswitch_procfs_exit
(
    void
)
{
    nvswitch_permissions_exit();

    if (!nvswitch_procfs_dir)
    {
        return;
    }

    NV_REMOVE_PROC_ENTRY(nvswitch_procfs_dir);
    nvswitch_procfs_dir = NULL;
}

static int
nvswitch_procfs_init
(
    void
)
{
    int rc = 0;

    if (!nvswitch_is_procfs_available)
    {
        return -EACCES;
    }

    nvswitch_procfs_dir = NV_CREATE_PROC_DIR(NVSWITCH_PROCFS_DIR, NULL);
    if (!nvswitch_procfs_dir)
    {
        return -EACCES;
    }

    rc = nvswitch_permissions_init();
    if (rc < 0)
    {
        goto cleanup;
    }

    return 0;

cleanup:

    nvswitch_procfs_exit();

    return rc;
}

static void
nvswitch_deinit_background_tasks
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    NV_ATOMIC_SET(nvswitch_dev->task_q_ready, 0);

    wake_up(&nvswitch_dev->wait_q_shutdown);

    nv_kthread_q_stop(&nvswitch_dev->task_q);
}

static int
nvswitch_init_background_tasks
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    int rc;

    rc = nv_kthread_q_init(&nvswitch_dev->task_q, nvswitch_dev->sname);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to create task queue\n", nvswitch_dev->name);
        return rc;
    }

    NV_ATOMIC_SET(nvswitch_dev->task_q_ready, 1);

    nv_kthread_q_item_init(&nvswitch_dev->task_item,
                           (nv_q_func_t) &nvswitch_task_dispatch,
                           nvswitch_dev);

    if (!nv_kthread_q_schedule_q_item(&nvswitch_dev->task_q,
                                      &nvswitch_dev->task_item))
    {
        printk(KERN_ERR "%s: Failed to schedule an item\n",nvswitch_dev->name);
        rc = -ENODEV;
        goto init_background_task_failed;
    }

    return 0;

init_background_task_failed:
    nvswitch_deinit_background_tasks(nvswitch_dev);

    return rc;
}

static NVSWITCH_DEV*
nvswitch_find_device(int minor)
{
    struct list_head *cur;
    NVSWITCH_DEV *nvswitch_dev = NULL;

    list_for_each(cur, &nvswitch.devices)
    {
        nvswitch_dev = list_entry(cur, NVSWITCH_DEV, list_node);
        if (nvswitch_dev->minor == minor)
        {
            return nvswitch_dev;
        }
    }

    return NULL;
}

static int
nvswitch_find_minor(void)
{
    struct list_head *cur;
    NVSWITCH_DEV *nvswitch_dev;
    int minor;
    int minor_in_use;

    for (minor = 0; minor < NVSWITCH_DEVICE_INSTANCE_MAX; minor++)
    {
        minor_in_use = 0;

        list_for_each(cur, &nvswitch.devices)
        {
            nvswitch_dev = list_entry(cur, NVSWITCH_DEV, list_node);
            if (nvswitch_dev->minor == minor)
            {
                minor_in_use = 1;
                break;
            }
        }

        if (!minor_in_use)
        {
            return minor;
        }
    }

    return NVSWITCH_DEVICE_INSTANCE_MAX;
}

static int
nvswitch_init_device
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    struct pci_dev *pci_dev = nvswitch_dev->pci_dev;
    NvlStatus retval;
    int rc;

    retval = nvswitch_lib_register_device(NV_PCI_DOMAIN_NUMBER(pci_dev),
                                          NV_PCI_BUS_NUMBER(pci_dev),
                                          NV_PCI_SLOT_NUMBER(pci_dev),
                                          PCI_FUNC(pci_dev->devfn),
                                          pci_dev->device,
                                          pci_dev,
                                          nvswitch_dev->minor,
                                          &nvswitch_dev->lib_device);
    if (NVL_SUCCESS != retval)
    {
        printk(KERN_ERR "%s: Failed to register device : %d\n",
               nvswitch_dev->name,
               retval);
        return -ENODEV;
    }

    nvswitch_load_bar_info(nvswitch_dev);

    retval = nvswitch_lib_initialize_device(nvswitch_dev->lib_device);
    if (NVL_SUCCESS != retval)
    {
        printk(KERN_ERR "%s: Failed to initialize device : %d\n",
               nvswitch_dev->name,
               retval);
        rc = -ENODEV;
        goto init_device_failed;
    }

    rc = nvswitch_initialize_device_interrupt(nvswitch_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to initialize interrupt : %d\n",
               nvswitch_dev->name,
               rc);
        goto init_intr_failed;
    }

    nvswitch_lib_enable_interrupts(nvswitch_dev->lib_device);

    return 0;

init_intr_failed:
    nvswitch_lib_shutdown_device(nvswitch_dev->lib_device);

init_device_failed:
    nvswitch_lib_unregister_device(nvswitch_dev->lib_device);
    nvswitch_dev->lib_device = NULL;

    return rc;
}

static void
nvswitch_post_init_device
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    nvswitch_lib_post_init_device(nvswitch_dev->lib_device);
}

static void
nvswitch_deinit_device
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    nvswitch_lib_disable_interrupts(nvswitch_dev->lib_device);

    nvswitch_shutdown_device_interrupt(nvswitch_dev);

    nvswitch_lib_shutdown_device(nvswitch_dev->lib_device);

    nvswitch_lib_unregister_device(nvswitch_dev->lib_device);
    nvswitch_dev->lib_device = NULL;
}

//
// Basic device open to support IOCTL interface
//
static int
nvswitch_device_open
(
    struct inode *inode,
    struct file *file
)
{
    NVSWITCH_DEV *nvswitch_dev;
    int rc = 0;

    //
    // Get the major/minor device
    // We might want this for routing requests to multiple nvswitches
    //
    printk(KERN_INFO "nvidia-nvswitch%d: open (major=%d)\n",
           MINOR(inode->i_rdev),
           MAJOR(inode->i_rdev));

    rc = mutex_lock_interruptible(&nvswitch.driver_mutex);
    if (rc)
    {
        return rc;
    }

    nvswitch_dev = nvswitch_find_device(MINOR(inode->i_rdev));
    if (!nvswitch_dev)
    {
        rc = -ENODEV;
        goto done;
    }

    NV_ATOMIC_INC(nvswitch_dev->ref_count);

    file->private_data = nvswitch_dev;

done:
    mutex_unlock(&nvswitch.driver_mutex);

    return rc;
}

//
// Basic device release to support IOCTL interface
//
static int
nvswitch_device_release
(
    struct inode *inode,
    struct file *file
)
{
    NVSWITCH_DEV *nvswitch_dev = file->private_data;

    printk(KERN_INFO "nvidia-nvswitch%d: release (major=%d)\n",
           MINOR(inode->i_rdev),
           MAJOR(inode->i_rdev));

    mutex_lock(&nvswitch.driver_mutex);

    file->private_data = NULL;

    //
    // If there are no outstanding references and the device is marked
    // unusable, free it.
    //
    if (NV_ATOMIC_DEC_AND_TEST(nvswitch_dev->ref_count) &&
        nvswitch_dev->unusable)
    {
        kfree(nvswitch_dev);
    }

    mutex_unlock(&nvswitch.driver_mutex);

    return 0;
}

static unsigned int
nvswitch_device_poll
(
    struct file *file,
    poll_table *wait
)
{
    NVSWITCH_DEV *nvswitch_dev = file->private_data;
    NvU32 nonfatal = 0;
    NvU32 fatal = 0;
    int rc = 0;

    rc = mutex_lock_interruptible(&nvswitch_dev->device_mutex);
    if (rc)
    {
        return rc;
    }

    if (nvswitch_dev->unusable)
    {
        printk(KERN_INFO "%s: a stale fd detected\n", nvswitch_dev->name);
        rc = POLLHUP;
        goto done;
    }

    poll_wait(file, &nvswitch_dev->wait_q_errors, wait);

    // Allow user to poll on either fatal, non-fatal or both types of errors.
    nvswitch_lib_get_log_count(nvswitch_dev->lib_device, &fatal, &nonfatal);
    rc = fatal ? POLLPRI : 0;
    rc |= nonfatal ? POLLIN : 0;

done:
    mutex_unlock(&nvswitch_dev->device_mutex);

    return rc;
}

typedef struct {
    void *kernel_params;                // Kernel copy of ioctl parameters
    unsigned long kernel_params_size;   // Size of ioctl params according to user
} IOCTL_STATE;

//
// Clean up any dynamically allocated memory for ioctl state
//
static void
nvswitch_ioctl_state_cleanup
(
    IOCTL_STATE *state
)
{
    kfree(state->kernel_params);
    state->kernel_params = NULL;
}

//
// Initialize buffer state for ioctl.
//
// This handles allocating memory and copying user data into kernel space.  The
// ioctl params structure only is supported. Nested data pointers are not handled.
//
// State is maintained in the IOCTL_STATE struct for use by the ioctl, _sync and
// _cleanup calls.
//
static int
nvswitch_ioctl_state_start(IOCTL_STATE *state, int cmd, unsigned long user_arg)
{
    int rc;

    state->kernel_params = NULL;
    state->kernel_params_size = _IOC_SIZE(cmd);

    if (0 == state->kernel_params_size)
    {
        return 0;
    }

    state->kernel_params = kzalloc(state->kernel_params_size, GFP_KERNEL);
    if (NULL == state->kernel_params)
    {
        rc = -ENOMEM;
        goto nvswitch_ioctl_state_start_fail;
    }

    // Copy params to kernel buffers.  Simple _IOR() ioctls can skip this step.
    if (_IOC_DIR(cmd) & _IOC_WRITE)
    {
        rc = copy_from_user(state->kernel_params,
                            (const void *)user_arg,
                            state->kernel_params_size);
        if (rc)
        {
            rc = -EFAULT;
            goto nvswitch_ioctl_state_start_fail;
        }
    }

    return 0;

nvswitch_ioctl_state_start_fail:
    nvswitch_ioctl_state_cleanup(state);
    return rc;
}

//
// Synchronize any ioctl output in the kernel buffers to the user mode buffers.
//
static int
nvswitch_ioctl_state_sync
(
    IOCTL_STATE *state,
    int cmd,
    unsigned long user_arg
)
{
    int rc;

    // Nothing to do if no buffer or write-only ioctl
    if ((0 == state->kernel_params_size) || (0 == (_IOC_DIR(cmd) & _IOC_READ)))
    {
        return 0;
    }

    // Copy params structure back to user mode
    rc = copy_to_user((void *)user_arg,
                      state->kernel_params,
                      state->kernel_params_size);
    if (rc)
    {
        rc = -EFAULT;
    }

    return rc;
}

static int
nvswitch_device_ioctl
(
    struct inode *inode,
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    NVSWITCH_DEV *nvswitch_dev = file->private_data;
    IOCTL_STATE state = {0};
    NvlStatus retval;
    int rc = 0;

    if (_IOC_TYPE(cmd) != NVSWITCH_DEV_IO_TYPE)
    {
        return -EINVAL;
    }

    rc = mutex_lock_interruptible(&nvswitch_dev->device_mutex);
    if (rc)
    {
        return rc;
    }

    if (nvswitch_dev->unusable)
    {
        printk(KERN_INFO "%s: a stale fd detected\n", nvswitch_dev->name);
        rc = -ENODEV;
        goto nvswitch_device_ioctl_exit;
    }

    rc = nvswitch_ioctl_state_start(&state, cmd, arg);
    if (rc)
    {
        goto nvswitch_device_ioctl_exit;
    }

    retval = nvswitch_lib_ctrl(nvswitch_dev->lib_device,
                               _IOC_NR(cmd),
                               state.kernel_params,
                               state.kernel_params_size);
    rc = nvswitch_map_status(retval);
    if (!rc)
    {
        rc = nvswitch_ioctl_state_sync(&state, cmd, arg);
    }

    nvswitch_ioctl_state_cleanup(&state);

nvswitch_device_ioctl_exit:
    mutex_unlock(&nvswitch_dev->device_mutex);

    return rc;
}

static long
nvswitch_device_unlocked_ioctl
(
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    return nvswitch_device_ioctl(NV_FILE_INODE(file), file, cmd, arg);
}

static int
nvswitch_ctl_check_version(NVSWITCH_CHECK_VERSION_PARAMS *p)
{
    NvlStatus retval;

    p->is_compatible = 0;

    retval = nvswitch_lib_check_api_version(p->user.major, p->user.minor,
                                            &p->kernel.major, &p->kernel.minor);
    if (retval == NVL_SUCCESS)
    {
        p->is_compatible = 1;
    }
    else if (retval == -NVL_ERR_NOT_SUPPORTED)
    {
        printk(KERN_ERR "nvidia-nvswitch: Version mismatch, "
               "kernel version %u.%u user version %u.%u\n",
                p->kernel.major, p->kernel.minor, p->user.major, p->user.minor);
    }
    else
    {
        // An unexpected failure
        return nvswitch_map_status(retval);
    }

    return 0;
}

static void
nvswitch_ctl_get_devices(NVSWITCH_GET_DEVICES_PARAMS *p)
{
    int index = 0;
    NVSWITCH_DEV *nvswitch_dev;
    struct list_head *cur;

    BUILD_BUG_ON(NVSWITCH_DEVICE_INSTANCE_MAX != NVSWITCH_MAX_DEVICES);

    list_for_each(cur, &nvswitch.devices)
    {
        nvswitch_dev = list_entry(cur, NVSWITCH_DEV, list_node);
        p->info[index].deviceInstance = nvswitch_dev->minor;
        p->info[index].pciDomain = NV_PCI_DOMAIN_NUMBER(nvswitch_dev->pci_dev);
        p->info[index].pciBus = NV_PCI_BUS_NUMBER(nvswitch_dev->pci_dev);
        p->info[index].pciDevice = NV_PCI_SLOT_NUMBER(nvswitch_dev->pci_dev);
        p->info[index].pciFunction = PCI_FUNC(nvswitch_dev->pci_dev->devfn);
        index++;
    }

    p->deviceCount = index;
}

#define NVSWITCH_CTL_CHECK_PARAMS(type, size) (sizeof(type) == size ? 0 : -EINVAL)

static int
nvswitch_ctl_cmd_dispatch
(
    unsigned int cmd,
    void *params,
    unsigned int param_size
)
{
    int rc;

    switch(cmd)
    {
        case CTRL_NVSWITCH_CHECK_VERSION:
            rc = NVSWITCH_CTL_CHECK_PARAMS(NVSWITCH_CHECK_VERSION_PARAMS,
                                           param_size);
            if (!rc)
            {
                rc = nvswitch_ctl_check_version(params);
            }
            break;
        case CTRL_NVSWITCH_GET_DEVICES:
            rc = NVSWITCH_CTL_CHECK_PARAMS(NVSWITCH_GET_DEVICES_PARAMS,
                                           param_size);
            if (!rc)
            {
                nvswitch_ctl_get_devices(params);
            }
            break;
        default:
            rc = -EINVAL;
            break;
    }

    return rc;
}

static int
nvswitch_ctl_ioctl
(
    struct inode *inode,
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    int rc = 0;
    IOCTL_STATE state = {0};

    if (_IOC_TYPE(cmd) != NVSWITCH_CTL_IO_TYPE)
    {
        return -EINVAL;
    }

    rc = mutex_lock_interruptible(&nvswitch.driver_mutex);
    if (rc)
    {
        return rc;
    }

    rc = nvswitch_ioctl_state_start(&state, cmd, arg);
    if (rc)
    {
        goto nvswitch_ctl_ioctl_exit;
    }

    rc = nvswitch_ctl_cmd_dispatch(_IOC_NR(cmd),
                                   state.kernel_params,
                                   state.kernel_params_size);
    if (!rc)
    {
        rc = nvswitch_ioctl_state_sync(&state, cmd, arg);
    }

    nvswitch_ioctl_state_cleanup(&state);

nvswitch_ctl_ioctl_exit:
    mutex_unlock(&nvswitch.driver_mutex);

    return rc;
}

static long
nvswitch_ctl_unlocked_ioctl
(
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    return nvswitch_ctl_ioctl(NV_FILE_INODE(file), file, cmd, arg);
}

static irqreturn_t
nvswitch_isr_pending
(
    int   irq,
    void *arg
)
{
    NVSWITCH_DEV *nvswitch_dev = (NVSWITCH_DEV *)arg;
    NvlStatus retval;

    //
    // On silicon MSI must be enabled.  Since interrupts will not be shared
    // with MSI, we can simply signal the thread.
    //
    if (nvswitch_dev->msi)
    {
        return IRQ_WAKE_THREAD;
    }

    //
    // The remainder of the function is a WAR for FSF.
    // TODO: Bug 1881361 to remove the FSF WAR
    //
    // We do not take mutex in the interrupt context. The interrupt
    // check is safe to driver state.
    //
    retval = nvswitch_lib_check_interrupts(nvswitch_dev->lib_device);

    // Wake interrupt thread if there is an interrupt pending
    if (-NVL_MORE_PROCESSING_REQUIRED == retval)
    {
        nvswitch_lib_disable_interrupts(nvswitch_dev->lib_device);
        return IRQ_WAKE_THREAD;
    }

    // PCI errors are handled else where.
    if (-NVL_PCI_ERROR == retval)
    {
        return IRQ_NONE;
    }

    if (NVL_SUCCESS != retval)
    {
        panic("nvidia-nvswitch: unrecoverable error in ISR\n");
    }

    return IRQ_NONE;
}

static irqreturn_t
nvswitch_isr_thread
(
    int   irq,
    void *arg
)
{
    NVSWITCH_DEV *nvswitch_dev = (NVSWITCH_DEV *)arg;
    NvlStatus retval;

    mutex_lock(&nvswitch_dev->device_mutex);

    retval = nvswitch_lib_service_interrupts(nvswitch_dev->lib_device);

    wake_up(&nvswitch_dev->wait_q_errors);

    //
    // This is part of the FSF non-MSI interrupt WAR. Interrupts were disabled in
    // in the ISR, so they must be re-enabled now.
    // TODO: Bug 1881361 to remove the FSF WAR
    //
    if (!nvswitch_dev->msi)
    {
        nvswitch_lib_enable_interrupts(nvswitch_dev->lib_device);
    }

    mutex_unlock(&nvswitch_dev->device_mutex);

    if (WARN_ON(retval != NVL_SUCCESS))
    {
        printk(KERN_ERR "%s: Interrupts disabled to avoid a storm\n",
               nvswitch_dev->name);
    }

    return IRQ_HANDLED;
}

static void
nvswitch_task_dispatch
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    NvU64 nsec;
    NvU64 timeout;
    NvS64 rc;

    if (NV_ATOMIC_READ(nvswitch_dev->task_q_ready) == 0)
    {
        return;
    }

    mutex_lock(&nvswitch_dev->device_mutex);

    nsec = nvswitch_lib_deferred_task_dispatcher(nvswitch_dev->lib_device);

    mutex_unlock(&nvswitch_dev->device_mutex);

    timeout = usecs_to_jiffies(nsec / NSEC_PER_USEC);

    rc = wait_event_interruptible_timeout(nvswitch_dev->wait_q_shutdown,
                              (NV_ATOMIC_READ(nvswitch_dev->task_q_ready) == 0),
                              timeout);

    //
    // These background tasks should rarely, if ever, get interrupted. We use
    // the "interruptible" variant of wait_event in order to avoid contributing
    // to the system load average (/proc/loadavg), and to avoid softlockup
    // warnings that can occur if a kernel thread lingers too long in an
    // uninterruptible state. If this does get interrupted, we'd like to debug
    // and find out why, so WARN in that case.
    //
    WARN_ON(rc < 0);

    //
    // Schedule a work item only if the above actually timed out or got
    // interrupted, without the condition becoming true.
    //
    if (rc <= 0)
    {
        if (!nv_kthread_q_schedule_q_item(&nvswitch_dev->task_q,
                                          &nvswitch_dev->task_item))
        {
            printk(KERN_ERR "%s: Failed to re-schedule background task\n",
                   nvswitch_dev->name);
        }
    }
}

static int
nvswitch_probe
(
    struct pci_dev *pci_dev,
    const struct pci_device_id *id_table
)
{
    NVSWITCH_DEV *nvswitch_dev = NULL;
    int rc = 0;
    int minor;

    if (!nvswitch_lib_validate_device_id(pci_dev->device))
    {
        return -EINVAL;
    }

    printk(KERN_INFO "nvidia-nvswitch: Probing device %04x:%02x:%02x.%x, "
           "Vendor Id = 0x%x, Device Id = 0x%x, Class = 0x%x \n",
           NV_PCI_DOMAIN_NUMBER(pci_dev),
           NV_PCI_BUS_NUMBER(pci_dev),
           NV_PCI_SLOT_NUMBER(pci_dev),
           PCI_FUNC(pci_dev->devfn),
           pci_dev->vendor,
           pci_dev->device,
           pci_dev->class);

    mutex_lock(&nvswitch.driver_mutex);

    minor = nvswitch_find_minor();
    if (minor >= NVSWITCH_DEVICE_INSTANCE_MAX)
    {
        rc = -ERANGE;
        goto find_minor_failed;
    }

    nvswitch_dev = kzalloc(sizeof(*nvswitch_dev), GFP_KERNEL);
    if (NULL == nvswitch_dev)
    {
        rc = -ENOMEM;
        goto kzalloc_failed;
    }

    mutex_init(&nvswitch_dev->device_mutex);
    init_waitqueue_head(&nvswitch_dev->wait_q_errors);
    init_waitqueue_head(&nvswitch_dev->wait_q_shutdown);

    snprintf(nvswitch_dev->name, sizeof(nvswitch_dev->name),
        NVSWITCH_DRIVER_NAME "%d", minor);

    snprintf(nvswitch_dev->sname, sizeof(nvswitch_dev->sname),
        NVSWITCH_SHORT_NAME "%d", minor);

    rc = pci_enable_device(pci_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to enable PCI device : %d\n",
               nvswitch_dev->name,
               rc);
        goto pci_enable_device_failed;
    }

    rc = pci_request_regions(pci_dev, nvswitch_dev->name);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to request memory regions : %d\n",
               nvswitch_dev->name,
               rc);
        goto pci_request_regions_failed;
    }

    nvswitch_dev->bar0 = pci_iomap(pci_dev, 0, 0);
    if (!nvswitch_dev->bar0)
    {
        rc = -ENOMEM;
        printk(KERN_ERR "%s: Failed to map BAR0 region : %d\n",
               nvswitch_dev->name,
               rc);
        goto pci_iomap_failed;
    }

    nvswitch_dev->pci_dev = pci_dev;
    nvswitch_dev->minor = minor;

    rc = nvswitch_init_device(nvswitch_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to initialize device : %d\n",
               nvswitch_dev->name,
               rc);
        goto init_device_failed;
    }

    nvswitch_post_init_device(nvswitch_dev);

    rc = nvswitch_init_background_tasks(nvswitch_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to initialize background tasks : %d\n",
               nvswitch_dev->name,
               rc);
        goto init_background_task_failed;
    }

    pci_set_drvdata(pci_dev, nvswitch_dev);

    list_add_tail(&nvswitch_dev->list_node, &nvswitch.devices);

    NV_ATOMIC_INC(nvswitch.count);

    mutex_unlock(&nvswitch.driver_mutex);

    return 0;

init_background_task_failed:
    nvswitch_deinit_device(nvswitch_dev);

init_device_failed:
    pci_iounmap(pci_dev, nvswitch_dev->bar0);

pci_iomap_failed:
    pci_release_regions(pci_dev);

pci_request_regions_failed:
    pci_disable_device(pci_dev);

pci_enable_device_failed:
    kfree(nvswitch_dev);

kzalloc_failed:
find_minor_failed:
    mutex_unlock(&nvswitch.driver_mutex);

    return rc;
}

void
nvswitch_remove
(
    struct pci_dev *pci_dev
)
{
    NVSWITCH_DEV *nvswitch_dev;

    mutex_lock(&nvswitch.driver_mutex);

    nvswitch_dev = pci_get_drvdata(pci_dev);

    if (nvswitch_dev == NULL)
    {
        goto done;
    }

    printk(KERN_INFO "%s: removing device %04x:%02x:%02x.%x\n",
           nvswitch_dev->name,
           NV_PCI_DOMAIN_NUMBER(pci_dev),
           NV_PCI_BUS_NUMBER(pci_dev),
           NV_PCI_SLOT_NUMBER(pci_dev),
           PCI_FUNC(pci_dev->devfn));

    //
    // Synchronize with device operations such as .ioctls/.poll, and then mark
    // the device unusable.
    //
    mutex_lock(&nvswitch_dev->device_mutex);
    nvswitch_dev->unusable = NV_TRUE;
    mutex_unlock(&nvswitch_dev->device_mutex);

    NV_ATOMIC_DEC(nvswitch.count);

    list_del(&nvswitch_dev->list_node);

    pci_set_drvdata(pci_dev, NULL);

    nvswitch_deinit_background_tasks(nvswitch_dev);

    nvswitch_deinit_device(nvswitch_dev);

    pci_iounmap(pci_dev, nvswitch_dev->bar0);

    pci_release_regions(pci_dev);

    pci_disable_device(pci_dev);

    // Free nvswitch_dev only if it is not in use.
    if (NV_ATOMIC_READ(nvswitch_dev->ref_count) == 0)
    {
        kfree(nvswitch_dev);
    }

done:
    mutex_unlock(&nvswitch.driver_mutex);

    return;
}

static void
nvswitch_load_bar_info
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    struct pci_dev *pci_dev = nvswitch_dev->pci_dev;
    nvlink_pci_info *info;
    NvU32 bar = 0;

    nvswitch_lib_get_device_info(nvswitch_dev->lib_device, &info);

    info->bars[0].offset = NVRM_PCICFG_BAR_OFFSET(0);
    pci_read_config_dword(pci_dev, info->bars[0].offset, &bar);

    info->bars[0].busAddress = (bar & PCI_BASE_ADDRESS_MEM_MASK);
    if (NV_PCI_RESOURCE_FLAGS(pci_dev, 0) & PCI_BASE_ADDRESS_MEM_TYPE_64)
    {
        pci_read_config_dword(pci_dev, info->bars[0].offset + 4, &bar);
        info->bars[0].busAddress |= (((NvU64)bar) << 32);
    }

    info->bars[0].baseAddr = NV_PCI_RESOURCE_START(pci_dev, 0);

    info->bars[0].barSize = NV_PCI_RESOURCE_SIZE(pci_dev, 0);

    info->bars[0].pBar = nvswitch_dev->bar0;
}

static int
nvswitch_initialize_device_interrupt
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    struct pci_dev *pci_dev = nvswitch_dev->pci_dev;
    int flags = 0;
    int rc;

#ifdef CONFIG_PCI_MSI
    rc = pci_enable_msi(pci_dev);
    if (0 == rc)
    {
        nvswitch_dev->msi = NV_TRUE;
    }
    else
#endif
    {
        nvswitch_dev->msi = NV_FALSE;
        flags |= IRQF_SHARED;
    }
    rc = request_threaded_irq(pci_dev->irq,
                              nvswitch_isr_pending,
                              nvswitch_isr_thread,
                              flags, nvswitch_dev->sname,
                              nvswitch_dev);
    if (rc)
    {
#ifdef CONFIG_PCI_MSI
        if (nvswitch_dev->msi)
            pci_disable_msi(pci_dev);
#endif
        printk(KERN_ERR "%s: failed to get IRQ (%d)\n",
               nvswitch_dev->name,
               nvswitch_dev->msi);

        return rc;
    }

    // Enable bus mastering on the device when MSI is enabled
    if (nvswitch_dev->msi)
    {
        pci_set_master(pci_dev);
    }

    return 0;
}

void
nvswitch_shutdown_device_interrupt
(
    NVSWITCH_DEV *nvswitch_dev
)
{
    struct pci_dev *pci_dev = nvswitch_dev->pci_dev;

    free_irq(pci_dev->irq, nvswitch_dev);
#ifdef CONFIG_PCI_MSI
    if (nvswitch_dev->msi)
        pci_disable_msi(pci_dev);
#endif
}

static void
nvswitch_ctl_exit
(
    void
)
{
    cdev_del(&nvswitch.cdev_ctl);
}

static int
nvswitch_ctl_init
(
    int major
)
{
    int rc = 0;
    dev_t nvswitch_ctl = MKDEV(major, NVSWITCH_CTL_MINOR);

    cdev_init(&nvswitch.cdev_ctl, &ctl_fops);

    nvswitch.cdev_ctl.owner = THIS_MODULE;

    rc = cdev_add(&nvswitch.cdev_ctl, nvswitch_ctl, 1);
    if (rc < 0)
    {
        printk(KERN_ERR "nvidia-nvswitch: Unable to create cdev ctl\n");
        return rc;
    }

    return 0;
}

//
// Initialize nvswitch driver SW state.  This is currently called
// from the RM as a backdoor interface, and not by the Linux device
// manager
//
int
nvswitch_init
(
    void
)
{
    int rc;

    if (nvswitch.initialized)
    {
        printk(KERN_ERR "nvidia-nvswitch: Interface already initialized\n");
        return -EBUSY;
    }

    BUILD_BUG_ON(NVSWITCH_DEVICE_INSTANCE_MAX >= NVSWITCH_MINOR_COUNT);

    mutex_init(&nvswitch.driver_mutex);

    INIT_LIST_HEAD(&nvswitch.devices);

    rc = alloc_chrdev_region(&nvswitch.devno,
                             0,
                             NVSWITCH_MINOR_COUNT,
                             NVSWITCH_DRIVER_NAME);
    if (rc < 0)
    {
        printk(KERN_ERR "nvidia-nvswitch: Unable to create cdev region\n");
        goto alloc_chrdev_region_fail;
    }

    printk(KERN_ERR, "nvidia-nvswitch: Major: %d Minor: %d\n",
           MAJOR(nvswitch.devno),
           MINOR(nvswitch.devno));

    cdev_init(&nvswitch.cdev, &device_fops);
    nvswitch.cdev.owner = THIS_MODULE;
    rc = cdev_add(&nvswitch.cdev, nvswitch.devno, NVSWITCH_DEVICE_INSTANCE_MAX);
    if (rc < 0)
    {
        printk(KERN_ERR "nvidia-nvswitch: Unable to create cdev\n");
        goto cdev_add_fail;
    }

    rc = pci_register_driver(&nvswitch_pci_driver);
    if (rc < 0)
    {
        printk(KERN_ERR "nvidia-nvswitch: Failed to register driver : %d\n", rc);
        goto pci_register_driver_fail;
    }

    rc = nvswitch_ctl_init(MAJOR(nvswitch.devno));
    if (rc < 0)
    {
        goto nvswitch_ctl_init_fail;
    }

    rc = nvswitch_procfs_init();
    if (rc < 0)
    {
        goto nvswitch_procfs_init_fail;
    }

    nvswitch.initialized = NV_TRUE;

    return 0;

nvswitch_procfs_init_fail:
    nvswitch_ctl_exit();

nvswitch_ctl_init_fail:
    pci_unregister_driver(&nvswitch_pci_driver);

pci_register_driver_fail:
    cdev_del(&nvswitch.cdev);

cdev_add_fail:
    unregister_chrdev_region(nvswitch.devno, NVSWITCH_MINOR_COUNT);

alloc_chrdev_region_fail:

    return rc;
}

//
// Clean up driver state on exit.  Currently called from RM backdoor call,
// and not by the Linux device manager.
//
void
nvswitch_exit
(
    void
)
{
    if (NV_FALSE == nvswitch.initialized)
    {
        return;
    }

    nvswitch_procfs_exit();

    nvswitch_ctl_exit();

    pci_unregister_driver(&nvswitch_pci_driver);

    cdev_del(&nvswitch.cdev);

    unregister_chrdev_region(nvswitch.devno, NVSWITCH_MINOR_COUNT);

    WARN_ON(!list_empty(&nvswitch.devices));

    nvswitch.initialized = NV_FALSE;
}

//
// Get current time in seconds.nanoseconds
// In this implementation, the time is from epoch time
// (midnight UTC of January 1, 1970)
//
NvU64
NVLINK_API_CALL
nvswitch_os_get_platform_time
(
    void
)
{
/*
 * Use ktime_get_real_ts64() instead of getnstimeofday(), when it is available.
 */
#if defined(NV_KTIME_GET_REAL_TS64_PRESENT)
    struct timespec64 ts;
    ktime_get_real_ts64(&ts);
    return ((NvU64) timespec64_to_ns(&ts));
#else
    struct timespec ts;
    getnstimeofday(&ts);
    return ((NvU64) timespec_to_ns(&ts));
#endif
}

void
NVLINK_API_CALL
nvswitch_os_print
(
    const int  log_level,
    const char *fmt,
    ...
)
{
    va_list arglist;
    char   *kern_level;
    char    fmt_printk[512];

    switch (log_level)
    {
        case NVSWITCH_DBG_LEVEL_MMIO:
            kern_level = KERN_DEBUG;
            break;
        case NVSWITCH_DBG_LEVEL_INFO:
            kern_level = KERN_INFO;
            break;
        case NVSWITCH_DBG_LEVEL_SETUP:
            kern_level = KERN_INFO;
            break;
        case NVSWITCH_DBG_LEVEL_WARN:
            kern_level = KERN_WARNING;
            break;
        case NVSWITCH_DBG_LEVEL_ERROR:
            kern_level = KERN_ERR;
            break;
        default:
            kern_level = KERN_DEFAULT;
            break;
    }

    va_start(arglist, fmt);
    snprintf(fmt_printk, sizeof(fmt_printk), "%s%s", kern_level, fmt);
    vprintk(fmt_printk, arglist);
    va_end(arglist);
}

void
NVLINK_API_CALL
nvswitch_os_override_platform
(
    void *os_handle,
    NvBool *rtlsim
)
{
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_read_registery_binary
(
    void *os_handle,
    const char *name,
    NvU8 *data,
    NvU32 length
)
{
    return -NVL_ERR_NOT_SUPPORTED;
}

NvU32
NVLINK_API_CALL
nvswitch_os_get_device_count
(
    void
)
{
    return NV_ATOMIC_READ(nvswitch.count);
}

//
// A helper to convert a string to an unsigned int.
//
// The string should be NULL terminated.
// Only works with base16 values.
//
static int
nvswitch_os_strtouint
(
    char *str,
    unsigned int *data
)
{
    char *p;
    unsigned long long val;

    if (!str || !data)
    {
        return -EINVAL;
    }

    *data = 0;
    val = 0;
    p = str;

    while (*p != '\0')
    {
        if ((tolower(*p) == 'x') && (*str == '0') && (p == str + 1))
        {
            p++;
        }
        else if (*p >='0' && *p <= '9')
        {
            val = val * 16 + (*p - '0');
            p++;
        }
        else if (tolower(*p) >= 'a' && tolower(*p) <= 'f')
        {
            val = val * 16 + (tolower(*p) - 'a' + 10);
            p++;
        }
        else
        {
            return -EINVAL;
        }
    }

    if (val > 0xFFFFFFFF)
    {
        return -EINVAL;
    }

    *data = (unsigned int)val;

    return 0;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_read_registry_dword
(
    void *os_handle,
    const char *name,
    NvU32 *data
)
{
    char *regkey, *regkey_val_start, *regkey_val_end;
    char regkey_val[NVSWITCH_REGKEY_VALUE_LEN + 1];
    NvU32 regkey_val_len = 0;

    *data = 0;

    if (!NvSwitchRegDwords)
    {
        return -NVL_ERR_GENERIC;
    }

    regkey = strstr(NvSwitchRegDwords, name);
    if (!regkey)
    {
        return -NVL_ERR_GENERIC;
    }

    regkey = strchr(regkey, '=');
    if (!regkey)
    {
        return -NVL_ERR_GENERIC;
    }

    regkey_val_start = regkey + 1;

    regkey_val_end = strchr(regkey, ';');
    if (!regkey_val_end)
    {
        regkey_val_end = strchr(regkey, '\0');
    }

    regkey_val_len = regkey_val_end - regkey_val_start;
    if (regkey_val_len > NVSWITCH_REGKEY_VALUE_LEN || regkey_val_len == 0)
    {
        return -NVL_ERR_GENERIC;
    }

    strncpy(regkey_val, regkey_val_start, regkey_val_len);
    regkey_val[regkey_val_len] = '\0';

    if (nvswitch_os_strtouint(regkey_val, data) != 0)
    {
        return -NVL_ERR_GENERIC;
    }

    return NVL_SUCCESS;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_alloc_contig_memory
(
    void **virt_addr,
    NvU32 size,
    NvBool force_dma32
)
{
    if (!virt_addr)
        return -NVL_BAD_ARGS;

    *virt_addr = kmalloc(PAGE_SIZE, GFP_KERNEL | force_dma32 ? GFP_DMA32 : 0);

    if(!*virt_addr)
    {
        pr_err("nvidia-nvswitch: unable to allocate kernel memory\n");
        return -NVL_NO_MEM;
    }

    return NVL_SUCCESS;
}

void
NVLINK_API_CALL
nvswitch_os_free_contig_memory
(
    void *virt_addr
)
{
    kfree(virt_addr);
}

static inline int
_nvswitch_to_pci_dma_direction
(
    NvU32 direction
)
{
    if (direction == NVSWITCH_DMA_DIR_TO_SYSMEM)
        return PCI_DMA_FROMDEVICE;
    else if (direction == NVSWITCH_DMA_DIR_FROM_SYSMEM)
        return PCI_DMA_TODEVICE;
    else
        return PCI_DMA_BIDIRECTIONAL;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_map_dma_region
(
    void *os_handle,
    void *cpu_addr,
    NvU64 *dma_handle,
    NvU32 size,
    NvU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev || !cpu_addr || !dma_handle)
        return -NVL_BAD_ARGS;

    dma_dir = _nvswitch_to_pci_dma_direction(direction);

    *dma_handle = (NvU64)pci_map_single(pdev, cpu_addr, size, dma_dir);

    if (pci_dma_mapping_error(pdev, *dma_handle))
    {
        pr_err("nvidia-nvswitch: unable to create PCI DMA mapping\n");
        return -NVL_ERR_GENERIC;
    }

    return NVL_SUCCESS;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_unmap_dma_region
(
    void *os_handle,
    void *cpu_addr,
    NvU64 dma_handle,
    NvU32 size,
    NvU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev || !cpu_addr)
        return -NVL_BAD_ARGS;

    dma_dir = _nvswitch_to_pci_dma_direction(direction);

    pci_unmap_single(pdev, dma_handle, size, dma_dir);

    return NVL_SUCCESS;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_set_dma_mask
(
    void *os_handle,
    NvU32 dma_addr_width
)
{
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev)
        return -NVL_BAD_ARGS;

    if (pci_set_dma_mask(pdev, DMA_BIT_MASK(dma_addr_width)))
        return -NVL_ERR_GENERIC;

    return NVL_SUCCESS;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_sync_dma_region_for_cpu
(
    void *os_handle,
    NvU64 dma_handle,
    NvU32 size,
    NvU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev)
        return -NVL_BAD_ARGS;

    dma_dir = _nvswitch_to_pci_dma_direction(direction);

    pci_dma_sync_single_for_cpu(pdev, dma_handle, size, dma_dir);

    return NVL_SUCCESS;
}

NvlStatus
NVLINK_API_CALL
nvswitch_os_sync_dma_region_for_device
(
    void *os_handle,
    NvU64 dma_handle,
    NvU32 size,
    NvU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev)
        return -NVL_BAD_ARGS;

    dma_dir = _nvswitch_to_pci_dma_direction(direction);

    pci_dma_sync_single_for_device(pdev, dma_handle, size, dma_dir);

    return NVL_SUCCESS;
}
