/*******************************************************************************
    Copyright (c) 2019 NVIDIA Corporation

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

#include "nv-linux.h"
#include "nv-caps.h"
#include "nv-procfs.h"

/*
 * This temporary module parameter will be nuked soon when devfs support
 * is enabled by default.
 */
static int nv_cap_enable_devfs = 1;
module_param(nv_cap_enable_devfs, int, S_IRUGO);
MODULE_PARM_DESC(nv_cap_enable_devfs, "Enable (1) or disable (0) nv-caps " \
                 "devfs support. Default: 1");

extern int NVreg_ModifyDeviceFiles;

/* sys_close() or __close_fd() */
#include <linux/syscalls.h>

#define NV_CAP_DRV_MINOR_COUNT 8192

typedef struct nv_cap_table_entry
{
    const char *name;
    int minor;
} nv_cap_table_entry_t;

#define NV_CAP_NUM_ENTRIES(_table) (sizeof(_table) / sizeof(_table[0]))

static nv_cap_table_entry_t g_nv_cap_nvlink_table[] =
{
    { "fabric-mgmt", -1 }
};

static nv_cap_table_entry_t g_nv_cap_mig_table[] =
{
    { "config", -1 },
    { "monitor", -1 }
};

#define NV_CAP_MIG_CI_ENTRIES(_gi) \
    { _gi"/ci0/access", -1 },      \
    { _gi"/ci1/access", -1 },      \
    { _gi"/ci2/access", -1 },      \
    { _gi"/ci3/access", -1 },      \
    { _gi"/ci4/access", -1 },      \
    { _gi"/ci5/access", -1 },      \
    { _gi"/ci6/access", -1 },      \
    { _gi"/ci7/access", -1 }

#define NV_CAP_MIG_GI_ENTRIES(_gpu) \
    { "gi0/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi0"),   \
    { "gi1/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi1"),   \
    { "gi2/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi2"),   \
    { "gi3/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi3"),   \
    { "gi4/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi4"),   \
    { "gi5/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi5"),   \
    { "gi6/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi6"),   \
    { "gi7/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi7"),   \
    { "gi8/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi8"),   \
    { "gi9/access", -1 },           \
    NV_CAP_MIG_CI_ENTRIES("gi9"),   \
    { "gi10/access", -1 },          \
    NV_CAP_MIG_CI_ENTRIES("gi10"),  \
    { "gi11/access", -1 },          \
    NV_CAP_MIG_CI_ENTRIES("gi11"),  \
    { "gi12/access", -1 },          \
    NV_CAP_MIG_CI_ENTRIES("gi12"),  \
    { "gi13/access", -1 },          \
    NV_CAP_MIG_CI_ENTRIES("gi13"),  \
    { "gi14/access", -1 },          \
    NV_CAP_MIG_CI_ENTRIES("gi14")

#define NV_CAP_MAX_GPUS 32

static nv_cap_table_entry_t g_nv_cap_mig_gpu_table[] =
{
    NV_CAP_MIG_GI_ENTRIES("gpu0"),
    NV_CAP_MIG_GI_ENTRIES("gpu1"),
    NV_CAP_MIG_GI_ENTRIES("gpu2"),
    NV_CAP_MIG_GI_ENTRIES("gpu3"),
    NV_CAP_MIG_GI_ENTRIES("gpu4"),
    NV_CAP_MIG_GI_ENTRIES("gpu5"),
    NV_CAP_MIG_GI_ENTRIES("gpu6"),
    NV_CAP_MIG_GI_ENTRIES("gpu7"),
    NV_CAP_MIG_GI_ENTRIES("gpu8"),
    NV_CAP_MIG_GI_ENTRIES("gpu9"),
    NV_CAP_MIG_GI_ENTRIES("gpu10"),
    NV_CAP_MIG_GI_ENTRIES("gpu11"),
    NV_CAP_MIG_GI_ENTRIES("gpu12"),
    NV_CAP_MIG_GI_ENTRIES("gpu13"),
    NV_CAP_MIG_GI_ENTRIES("gpu14"),
    NV_CAP_MIG_GI_ENTRIES("gpu15"),
    NV_CAP_MIG_GI_ENTRIES("gpu16"),
    NV_CAP_MIG_GI_ENTRIES("gpu17"),
    NV_CAP_MIG_GI_ENTRIES("gpu18"),
    NV_CAP_MIG_GI_ENTRIES("gpu19"),
    NV_CAP_MIG_GI_ENTRIES("gpu20"),
    NV_CAP_MIG_GI_ENTRIES("gpu21"),
    NV_CAP_MIG_GI_ENTRIES("gpu22"),
    NV_CAP_MIG_GI_ENTRIES("gpu23"),
    NV_CAP_MIG_GI_ENTRIES("gpu24"),
    NV_CAP_MIG_GI_ENTRIES("gpu25"),
    NV_CAP_MIG_GI_ENTRIES("gpu26"),
    NV_CAP_MIG_GI_ENTRIES("gpu27"),
    NV_CAP_MIG_GI_ENTRIES("gpu28"),
    NV_CAP_MIG_GI_ENTRIES("gpu29"),
    NV_CAP_MIG_GI_ENTRIES("gpu30"),
    NV_CAP_MIG_GI_ENTRIES("gpu31")
};

struct nv_cap
{
    char *path;
    char *name;
    int minor;
    int permissions;
    int modify;
    struct proc_dir_entry *parent;
    struct proc_dir_entry *entry;
};

#define NV_CAP_PROCFS_WRITE_BUF_SIZE 128

typedef struct nv_cap_file_private
{
    int minor;
    int permissions;
    /* Will be used for fork() detection.*/
    struct files_struct *files;
    int modify;
    char buffer[NV_CAP_PROCFS_WRITE_BUF_SIZE];
    off_t offset;
} nv_cap_file_private_t;

struct
{
    NvBool initialized;
    struct cdev cdev;
    dev_t devno;
} g_nv_cap_drv;

#define NV_CAP_PROCFS_DIR "driver/nvidia-caps"

static struct proc_dir_entry *nv_cap_procfs_dir;
static struct proc_dir_entry *nv_cap_procfs_nvlink_minors;
static struct proc_dir_entry *nv_cap_procfs_mig_minors;

static int nv_procfs_read_nvlink_minors(struct seq_file *s, void *v)
{
    int i, count;

    count = NV_CAP_NUM_ENTRIES(g_nv_cap_nvlink_table);
    for (i = 0; i < count; i++)
    {
        seq_printf(s, "%s %d\n", g_nv_cap_nvlink_table[i].name,
                   g_nv_cap_nvlink_table[i].minor);
    }

    return 0;
}

static int nv_procfs_read_mig_minors(struct seq_file *s, void *v)
{
    int i, j, count, index;

    count = NV_CAP_NUM_ENTRIES(g_nv_cap_mig_table);
    for (i = 0; i < count; i++)
    {
        seq_printf(s, "%s %d\n", g_nv_cap_mig_table[i].name,
                   g_nv_cap_mig_table[i].minor);
    }

    index = 0;
    count = NV_CAP_NUM_ENTRIES(g_nv_cap_mig_gpu_table) / NV_CAP_MAX_GPUS;
    for (i = 0; i < NV_CAP_MAX_GPUS; i++)
    {
        for (j = 0; j < count; j++)
        {
            seq_printf(s, "gpu%d/%s %d\n", i,
                       g_nv_cap_mig_gpu_table[index].name,
                       g_nv_cap_mig_gpu_table[index].minor);
            index++;
        }
    }

    return 0;
}

NV_DEFINE_SINGLE_PROCFS_FILE(nvlink_minors,
                             NV_READ_LOCK_SYSTEM_PM_LOCK_INTERRUPTIBLE,
                             NV_READ_UNLOCK_SYSTEM_PM_LOCK);

NV_DEFINE_SINGLE_PROCFS_FILE(mig_minors,
                             NV_READ_LOCK_SYSTEM_PM_LOCK_INTERRUPTIBLE,
                             NV_READ_UNLOCK_SYSTEM_PM_LOCK);

static void nv_cap_procfs_exit(void)
{
    if (!nv_cap_procfs_dir)
    {
        return;
    }

    nv_procfs_unregister_all(nv_cap_procfs_dir, nv_cap_procfs_dir);
    nv_cap_procfs_dir = NULL;
}

int nv_cap_procfs_init(void)
{
    nv_cap_procfs_dir = NV_CREATE_PROC_DIR(NV_CAP_PROCFS_DIR, NULL);
    if (nv_cap_procfs_dir == NULL)
    {
        return -EACCES;
    }

    nv_cap_procfs_mig_minors = NV_CREATE_PROC_FILE("mig-minors",
                                                   nv_cap_procfs_dir,
                                                   mig_minors,
                                                   NULL);
    if (nv_cap_procfs_mig_minors == NULL)
    {
        goto cleanup;
    }

    nv_cap_procfs_nvlink_minors = NV_CREATE_PROC_FILE("nvlink-minors",
                                                      nv_cap_procfs_dir,
                                                      nvlink_minors,
                                                      NULL);
    if (nv_cap_procfs_nvlink_minors == NULL)
    {
        goto cleanup;
    }

    return 0;

cleanup:
    nv_cap_procfs_exit();

    return -EACCES;
}

static int nv_cap_find_minor_in_table(nv_cap_table_entry_t *table, int count,
                                      const char *target)
{
    int i;

    for (i = 0; i < count; i++)
    {
        if (strcmp(table[i].name, target) == 0)
        {
            return table[i].minor;
        }
    }

    return -1;
}

static int nv_cap_find_minor(char *path)
{
    char *target;
    int minor;
    unsigned int len, count, gpu, index;

    target = "/driver/nvidia-nvlink/capabilities/";
    len = strlen(target);

    if (strncmp(path, target, len) == 0)
    {
        target = path + len;
        count = NV_CAP_NUM_ENTRIES(g_nv_cap_nvlink_table);
        minor = nv_cap_find_minor_in_table(g_nv_cap_nvlink_table,
                                           count, target);
        if (minor >= 0)
        {
            return minor;
        }
    }

    target = "/driver/nvidia/capabilities/gpu";
    len = strlen(target);

    if (strncmp(path, target, len) == 0)
    {
        target = path + len;
        sscanf(target, "%u\n", &gpu);
        target = strchr(target, '/');
        WARN_ON(gpu >= NV_CAP_MAX_GPUS);
        WARN_ON(target == NULL);

        len = strlen("/mig/");
        if (strncmp(target, "/mig/", len) == 0)
        {
            target = target + len;
            count = NV_CAP_NUM_ENTRIES(g_nv_cap_mig_gpu_table) / NV_CAP_MAX_GPUS;
            index = gpu * count;
            minor = nv_cap_find_minor_in_table(&g_nv_cap_mig_gpu_table[index],
                                               count, target);
            if (minor >= 0)
            {
                return minor;
            }
        }
    }

    target = "/driver/nvidia/capabilities/mig/";
    len = strlen(target);

    if (strncmp(path, target, len) == 0)
    {
        target = path + len;

        count = NV_CAP_NUM_ENTRIES(g_nv_cap_mig_table);
        minor = nv_cap_find_minor_in_table(g_nv_cap_mig_table, count, target);
        if (minor >= 0)
        {
            return minor;
        }
    }

    return -1;
}

static void nv_cap_assign_minors(void)
{
    int minor = 0;
    int count;
    int i;

    count = NV_CAP_NUM_ENTRIES(g_nv_cap_nvlink_table);
    for (i = 0; i < count; i++)
    {
        g_nv_cap_nvlink_table[i].minor = minor++;
    }

    count = NV_CAP_NUM_ENTRIES(g_nv_cap_mig_table);
    for (i = 0; i < count; i++)
    {
        g_nv_cap_mig_table[i].minor = minor++;
    }

    count = NV_CAP_NUM_ENTRIES(g_nv_cap_mig_gpu_table);
    for (i = 0; i < count; i++)
    {
        g_nv_cap_mig_gpu_table[i].minor = minor++;
    }

    WARN_ON(minor > NV_CAP_DRV_MINOR_COUNT);
}

static ssize_t nv_cap_procfs_write(struct file *file,
                                    const char __user *buffer,
                                    size_t count, loff_t *pos)
{
    nv_cap_file_private_t *private = NULL;
    unsigned long bytes_left;
    char *proc_buffer;

    private = ((struct seq_file *)file->private_data)->private;
    bytes_left = (sizeof(private->buffer) - private->offset - 1);

    if (count == 0)
    {
        return -EINVAL;
    }

    if ((bytes_left == 0) || (count > bytes_left))
    {
        return -ENOSPC;
    }

    proc_buffer = &private->buffer[private->offset];

    if (copy_from_user(proc_buffer, buffer, count))
    {
        nv_printf(NV_DBG_ERRORS, "nv-caps: failed to copy in proc data!\n");
        return -EFAULT;
    }

    private->offset += count;
    proc_buffer[count] = '\0';

    *pos = private->offset;

    return count;
}

static int nv_cap_procfs_read(struct seq_file *s, void *v)
{
    nv_cap_file_private_t *private = s->private;

    if (nv_cap_enable_devfs)
    {
        seq_printf(s, "%s: %d\n", "DeviceFileMinor", private->minor);
        seq_printf(s, "%s: %d\n", "DeviceFileMode", private->permissions);
        seq_printf(s, "%s: %d\n", "DeviceFileModify", private->modify);
    }

    return 0;
}

static int nv_cap_procfs_open(struct inode *inode, struct file *file)
{
    nv_cap_file_private_t *private = NULL;
    int rc;
    nv_cap_t *cap = NV_PDE_DATA(inode);

    NV_KMALLOC(private, sizeof(nv_cap_file_private_t));
    if (private == NULL)
    {
        return -ENOMEM;
    }

    /* Just copy over data for "fd" validation */
    private->files = current->files;
    private->minor = cap->minor;
    private->permissions = cap->permissions;
    private->offset = 0;
    private->modify = cap->modify;

    rc = single_open(file, nv_cap_procfs_read, private);
    if (rc < 0)
    {
        NV_KFREE(private, sizeof(nv_cap_file_private_t));
        return rc;
    }

    rc = NV_READ_LOCK_SYSTEM_PM_LOCK_INTERRUPTIBLE();
    if (rc < 0)
    {
        single_release(inode, file);
        NV_KFREE(private, sizeof(nv_cap_file_private_t));
    }

    return rc;
}

static int nv_cap_procfs_release(struct inode *inode, struct file *file)
{
    struct seq_file *s = file->private_data;
    nv_cap_file_private_t *private = NULL;
    char *buffer;
    int modify;
    nv_cap_t *cap = NV_PDE_DATA(inode);

    if (s != NULL)
    {
        private = s->private;
    }

    NV_READ_UNLOCK_SYSTEM_PM_LOCK();

    single_release(inode, file);

    if (private != NULL)
    {
        buffer = private->buffer;

        if (private->offset != 0)
        {
            if (sscanf(buffer, "DeviceFileModify: %d", &modify) == 1)
            {
                cap->modify = modify;
            }
        }

        NV_KFREE(private, sizeof(nv_cap_file_private_t));
    }

    /*
     * All open files using the proc entry will be invalidated
     * if the entry is removed.
     */
    file->private_data = NULL;

    return 0;
}

static nv_proc_ops_t g_nv_cap_procfs_fops = {
    NV_PROC_OPS_SET_OWNER()
    .NV_PROC_OPS_OPEN    = nv_cap_procfs_open,
    .NV_PROC_OPS_RELEASE = nv_cap_procfs_release,
    .NV_PROC_OPS_WRITE   = nv_cap_procfs_write,
    .NV_PROC_OPS_READ    = seq_read,
    .NV_PROC_OPS_LSEEK   = seq_lseek,
};

/* forward declaration of g_nv_cap_drv_fops */
static struct file_operations g_nv_cap_drv_fops;

int NV_API_CALL nv_cap_validate_and_dup_fd(const nv_cap_t *cap, int fd)
{
    struct file *file;
    int dup_fd;
    const nv_cap_file_private_t *private = NULL;
    struct inode *inode = NULL;
    dev_t rdev = 0;

    if (cap == NULL)
    {
        return -1;
    }

    file = fget(fd);
    if (file == NULL)
    {
        return -1;
    }

    inode = NV_FILE_INODE(file);
    if (inode == NULL)
    {
        goto err;
    }

    if (nv_cap_enable_devfs)
    {
        /* Make sure the fd belongs to the nv-cap-drv */
        if (file->f_op != &g_nv_cap_drv_fops)
        {
            goto err;
        }

        /* Make sure the fd has the expected capability */
        rdev = inode->i_rdev;
        if (MINOR(rdev) != cap->minor)
        {
            goto err;
        }

        private = file->private_data;
    }
    else
    {
        /*
         * Make sure the fd belongs to procfs and validate the associated
         * capability; the NV_PDE_DATA(inode) check ensures this file was created
         * by nv_cap_create_file_entry().
         */
        if ((strcmp(inode->i_sb->s_type->name, "proc") != 0) ||
            (NV_PDE_DATA(inode) != cap))
        {
            goto err;
        }

        private = ((struct seq_file *)file->private_data)->private;
    }

    /* Make sure the fd is not duped during fork() */
    if ((private == NULL) ||
        (private->files != current->files))
    {
        goto err;
    }

    dup_fd = NV_GET_UNUSED_FD_FLAGS(O_CLOEXEC);
    if (dup_fd < 0)
    {
          dup_fd = NV_GET_UNUSED_FD();
          if (dup_fd < 0)
          {
               goto err;
          }
    }

    fd_install(dup_fd, file);
    return dup_fd;

err:
    fput(file);
    return -1;
}

void NV_API_CALL nv_cap_close_fd(int fd)
{
    if (fd == -1)
    {
        return;
    }

    /*
     * Acquire task_lock as we access current->files explicitly (__close_fd)
     * and implicitly (sys_close), and it will race with the exit path.
     */
    task_lock(current);

    /* Nothing to do, we are in exit path */
    if (current->files == NULL)
    {
        task_unlock(current);
        return;
    }

/*
 * From v4.17-rc1 kernels have stopped exporting sys_close(fd) and started
 * exporting __close_fd, as of this commit:
 * 2018-04-02 2ca2a09d6215 ("fs: add ksys_close() wrapper; remove in-kernel
 *  calls to sys_close()")
 */
#if NV_IS_EXPORT_SYMBOL_PRESENT___close_fd
    __close_fd(current->files, fd);
#else
    sys_close(fd);
#endif

    task_unlock(current);
}

static nv_cap_t* nv_cap_alloc(nv_cap_t *parent_cap, const char *name)
{
    nv_cap_t *cap;
    int len;

    if (parent_cap == NULL || name == NULL)
    {
        return NULL;
    }

    NV_KMALLOC(cap, sizeof(nv_cap_t));
    if (cap == NULL)
    {
        return NULL;
    }

    len = strlen(name) + strlen(parent_cap->path) + 2;
    NV_KMALLOC(cap->path, len);
    if (cap->path == NULL)
    {
        NV_KFREE(cap, sizeof(nv_cap_t));
        return NULL;
    }

    strcpy(cap->path, parent_cap->path);
    strcat(cap->path, "/");
    strcat(cap->path, name);

    len = strlen(name) + 1;
    NV_KMALLOC(cap->name, len);
    if (cap->name == NULL)
    {
        NV_KFREE(cap->path, strlen(cap->path) + 1);
        NV_KFREE(cap, sizeof(nv_cap_t));
        return NULL;
    }

    strcpy(cap->name, name);

    cap->minor = -1;
    cap->modify = NVreg_ModifyDeviceFiles;

    return cap;
}

static void nv_cap_free(nv_cap_t *cap)
{
    if (cap == NULL)
    {
        return;
    }

    NV_KFREE(cap->path, strlen(cap->path) + 1);
    NV_KFREE(cap->name, strlen(cap->name) + 1);
    NV_KFREE(cap, sizeof(nv_cap_t));
}

nv_cap_t* NV_API_CALL nv_cap_create_file_entry(nv_cap_t *parent_cap,
                                               const char *name, int mode)
{
    nv_cap_t *cap = NULL;
    int minor;

    cap = nv_cap_alloc(parent_cap, name);
    if (cap == NULL)
    {
        return NULL;
    }

    cap->parent = parent_cap->entry;
    cap->permissions = mode;

    /* Make proc entries world visible if devfs is enabled */
    if (nv_cap_enable_devfs)
    {
        mode = (S_IFREG | S_IRUGO);
    }
    else
    {
        mode |= S_IFREG;
    }

    minor = nv_cap_find_minor(cap->path);
    if (minor < 0)
    {
        nv_cap_free(cap);
        return NULL;
    }

    cap->minor = minor;

    cap->entry = proc_create_data(name, mode, parent_cap->entry,
                                  &g_nv_cap_procfs_fops, (void*)cap);
    if (cap->entry == NULL)
    {
        nv_cap_free(cap);
        return NULL;
    }

    return cap;
}

nv_cap_t* NV_API_CALL nv_cap_create_dir_entry(nv_cap_t *parent_cap,
                                              const char *name, int mode)
{
    nv_cap_t *cap = NULL;

    cap = nv_cap_alloc(parent_cap, name);
    if (cap == NULL)
    {
        return NULL;
    }

    cap->parent = parent_cap->entry;
    cap->permissions = mode;
    cap->minor = -1;

    /* Make proc entries world visible if devfs is enabled */
    if (nv_cap_enable_devfs)
    {
        mode = (S_IFDIR | S_IRUGO | S_IXUGO);
    }
    else
    {
        mode |= S_IFDIR;
    }

    cap->entry = NV_PROC_MKDIR_MODE(name, mode, parent_cap->entry);
    if (cap->entry == NULL)
    {
        nv_cap_free(cap);
        return NULL;
    }

    return cap;
}

nv_cap_t* NV_API_CALL nv_cap_init(const char *path)
{
    nv_cap_t parent_cap;
    nv_cap_t *cap;
    int mode;
    char *name = NULL;
    char dir[] = "/capabilities";

    if (path == NULL)
    {
        return NULL;
    }

    NV_KMALLOC(name, (strlen(path) + strlen(dir)) + 1);
    if (name == NULL)
    {
        return NULL;
    }

    strcpy(name, path);
    strcat(name, dir);
    parent_cap.entry = NULL;
    parent_cap.path = "";
    parent_cap.name = "";
    mode =  S_IRUGO | S_IXUGO;
    cap = nv_cap_create_dir_entry(&parent_cap, name, mode);

    NV_KFREE(name, strlen(name) + 1);
    return cap;
}

void NV_API_CALL nv_cap_destroy_entry(nv_cap_t *cap)
{
    if (WARN_ON(cap == NULL))
    {
        return;
    }

    remove_proc_entry(cap->name, cap->parent);
    nv_cap_free(cap);
}

static int nv_cap_drv_open(struct inode *inode, struct file *file)
{
    nv_cap_file_private_t *private = NULL;

    NV_KMALLOC(private, sizeof(nv_cap_file_private_t));
    if (private == NULL)
    {
        return -ENOMEM;
    }

    /* Just copy over data for "fd" validation */
    private->files = current->files;
    file->private_data = private;

    return 0;
}

static int nv_cap_drv_release(struct inode *inode, struct file *file)
{
    if (file->private_data != NULL)
    {
        NV_KFREE(file->private_data, sizeof(nv_cap_file_private_t));

        file->private_data = NULL;
    }

    return 0;
}

static struct file_operations g_nv_cap_drv_fops =
{
    .owner = THIS_MODULE,
    .open    = nv_cap_drv_open,
    .release = nv_cap_drv_release
};

int NV_API_CALL nv_cap_drv_init(void)
{
    int rc;

    nv_cap_assign_minors();

    if (!nv_cap_enable_devfs)
    {
        nv_printf(NV_DBG_INFO, "nv-caps-drv is disabled.\n");
        return 0;
    }

    if (g_nv_cap_drv.initialized)
    {
        nv_printf(NV_DBG_ERRORS, "nv-caps-drv is already initialized.\n");
        return -EBUSY;
    }

    rc = alloc_chrdev_region(&g_nv_cap_drv.devno,
                             0,
                             NV_CAP_DRV_MINOR_COUNT,
                             "nvidia-caps");
    if (rc < 0)
    {
        nv_printf(NV_DBG_ERRORS, "nv-caps-drv failed to create cdev region.\n");
        return rc;
    }

    cdev_init(&g_nv_cap_drv.cdev, &g_nv_cap_drv_fops);

    g_nv_cap_drv.cdev.owner = THIS_MODULE;

    rc = cdev_add(&g_nv_cap_drv.cdev, g_nv_cap_drv.devno,
                  NV_CAP_DRV_MINOR_COUNT);
    if (rc < 0)
    {
        nv_printf(NV_DBG_ERRORS, "nv-caps-drv failed to create cdev.\n");
        goto cdev_add_fail;
    }

    rc = nv_cap_procfs_init();
    if (rc < 0)
    {
        nv_printf(NV_DBG_ERRORS, "nv-caps-drv: unable to init proc\n");
        goto proc_init_fail;
    }

    g_nv_cap_drv.initialized = NV_TRUE;

    return 0;

proc_init_fail:
    cdev_del(&g_nv_cap_drv.cdev);

cdev_add_fail:
    unregister_chrdev_region(g_nv_cap_drv.devno, NV_CAP_DRV_MINOR_COUNT);

    return rc;
}

void NV_API_CALL nv_cap_drv_exit(void)
{
    if (!g_nv_cap_drv.initialized)
    {
        return;
    }

    nv_cap_procfs_exit();

    cdev_del(&g_nv_cap_drv.cdev);

    unregister_chrdev_region(g_nv_cap_drv.devno, NV_CAP_DRV_MINOR_COUNT);

    g_nv_cap_drv.initialized = NV_FALSE;
}
