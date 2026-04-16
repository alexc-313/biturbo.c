#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_reserved_mem.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "../biturbo_cma_ioctl.h"

#define BITURBO_CMA_MAX_BUFS 1024

struct biturbo_cma_buf {
    bool in_use;
    struct file *owner;
    void *cpu_addr;
    dma_addr_t dma_addr;
    size_t size;
    u64 mmap_offset;
    char name[BITURBO_CMA_NAME_LEN];
};

struct biturbo_cma_dev {
    struct device *dev;
    struct miscdevice miscdev;
    struct mutex lock;
    struct biturbo_cma_buf bufs[BITURBO_CMA_MAX_BUFS];
};

static void biturbo_cma_free_one(struct biturbo_cma_dev *bcdev, unsigned int idx)
{
    struct biturbo_cma_buf *buf;

    if (!bcdev || idx >= BITURBO_CMA_MAX_BUFS)
        return;

    buf = &bcdev->bufs[idx];
    if (!buf->in_use)
        return;

    dma_free_coherent(bcdev->dev, buf->size, buf->cpu_addr, buf->dma_addr);
    memset(buf, 0, sizeof(*buf));
}

static int biturbo_cma_open(struct inode *inode, struct file *file)
{
    struct miscdevice *misc = file->private_data;
    struct biturbo_cma_dev *bcdev;

    bcdev = container_of(misc, struct biturbo_cma_dev, miscdev);
    file->private_data = bcdev;
    return 0;
}

static int biturbo_cma_release(struct inode *inode, struct file *file)
{
    struct biturbo_cma_dev *bcdev = file->private_data;
    unsigned int i;

    if (!bcdev)
        return 0;

    mutex_lock(&bcdev->lock);
    for (i = 0; i < BITURBO_CMA_MAX_BUFS; i++) {
        if (bcdev->bufs[i].in_use && bcdev->bufs[i].owner == file)
            biturbo_cma_free_one(bcdev, i);
    }
    mutex_unlock(&bcdev->lock);
    return 0;
}

static long biturbo_cma_ioctl_alloc(struct biturbo_cma_dev *bcdev,
                                    struct file *file,
                                    unsigned long arg)
{
    biturbo_cma_alloc_req_t req;
    struct biturbo_cma_buf *buf = NULL;
    size_t alloc_size;
    unsigned int i;
    void *cpu_addr;
    dma_addr_t dma_addr;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    if (req.size == 0)
        return -EINVAL;

    alloc_size = PAGE_ALIGN((size_t)req.size);

    mutex_lock(&bcdev->lock);
    for (i = 0; i < BITURBO_CMA_MAX_BUFS; i++) {
        if (!bcdev->bufs[i].in_use) {
            buf = &bcdev->bufs[i];
            break;
        }
    }
    if (!buf) {
        mutex_unlock(&bcdev->lock);
        return -ENOSPC;
    }

    cpu_addr = dma_alloc_coherent(bcdev->dev, alloc_size, &dma_addr, GFP_KERNEL);
    if (!cpu_addr) {
        mutex_unlock(&bcdev->lock);
        return -ENOMEM;
    }

    memset(buf, 0, sizeof(*buf));
    buf->in_use = true;
    buf->owner = file;
    buf->cpu_addr = cpu_addr;
    buf->dma_addr = dma_addr;
    buf->size = alloc_size;
    buf->mmap_offset = ((u64)i + 1ULL) << PAGE_SHIFT;
    strlcpy(buf->name, req.name, sizeof(buf->name));

    req.handle = i;
    req.size = alloc_size;
    req.dma_addr = (u64)dma_addr;
    req.mmap_offset = buf->mmap_offset;

    mutex_unlock(&bcdev->lock);

    if (copy_to_user((void __user *)arg, &req, sizeof(req))) {
        mutex_lock(&bcdev->lock);
        biturbo_cma_free_one(bcdev, req.handle);
        mutex_unlock(&bcdev->lock);
        return -EFAULT;
    }

    return 0;
}

static long biturbo_cma_ioctl_free(struct biturbo_cma_dev *bcdev,
                                   struct file *file,
                                   unsigned long arg)
{
    biturbo_cma_free_req_t req;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;
    if (req.handle >= BITURBO_CMA_MAX_BUFS)
        return -EINVAL;

    mutex_lock(&bcdev->lock);
    if (!bcdev->bufs[req.handle].in_use) {
        mutex_unlock(&bcdev->lock);
        return -ENOENT;
    }
    if (bcdev->bufs[req.handle].owner != file) {
        mutex_unlock(&bcdev->lock);
        return -EPERM;
    }

    biturbo_cma_free_one(bcdev, req.handle);
    mutex_unlock(&bcdev->lock);
    return 0;
}

static long biturbo_cma_unlocked_ioctl(struct file *file,
                                       unsigned int cmd,
                                       unsigned long arg)
{
    struct biturbo_cma_dev *bcdev = file->private_data;

    switch (cmd) {
    case BITURBO_CMA_IOCTL_ALLOC:
        return biturbo_cma_ioctl_alloc(bcdev, file, arg);
    case BITURBO_CMA_IOCTL_FREE:
        return biturbo_cma_ioctl_free(bcdev, file, arg);
    default:
        return -ENOTTY;
    }
}

static int biturbo_cma_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct biturbo_cma_dev *bcdev = file->private_data;
    struct biturbo_cma_buf *buf = NULL;
    unsigned long saved_pgoff = vma->vm_pgoff;
    size_t vma_size = (size_t)(vma->vm_end - vma->vm_start);
    unsigned int i;
    int ret;

    mutex_lock(&bcdev->lock);
    for (i = 0; i < BITURBO_CMA_MAX_BUFS; i++) {
        if (bcdev->bufs[i].in_use &&
            bcdev->bufs[i].owner == file &&
            bcdev->bufs[i].mmap_offset == ((u64)saved_pgoff << PAGE_SHIFT)) {
            buf = &bcdev->bufs[i];
            break;
        }
    }

    if (!buf) {
        mutex_unlock(&bcdev->lock);
        return -ENOENT;
    }
    if (vma_size > buf->size) {
        mutex_unlock(&bcdev->lock);
        return -EINVAL;
    }

    vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
    vma->vm_pgoff = 0;
    ret = dma_mmap_coherent(bcdev->dev, vma, buf->cpu_addr,
                            buf->dma_addr, buf->size);
    vma->vm_pgoff = saved_pgoff;
    mutex_unlock(&bcdev->lock);
    return ret;
}

static const struct file_operations biturbo_cma_fops = {
    .owner          = THIS_MODULE,
    .open           = biturbo_cma_open,
    .release        = biturbo_cma_release,
    .unlocked_ioctl = biturbo_cma_unlocked_ioctl,
    .mmap           = biturbo_cma_mmap,
    .llseek         = no_llseek,
};

static int biturbo_cma_probe(struct platform_device *pdev)
{
    struct biturbo_cma_dev *bcdev;
    int ret;

    bcdev = devm_kzalloc(&pdev->dev, sizeof(*bcdev), GFP_KERNEL);
    if (!bcdev)
        return -ENOMEM;

    bcdev->dev = &pdev->dev;
    mutex_init(&bcdev->lock);

    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    if (ret)
        return ret;

    ret = of_reserved_mem_device_init(&pdev->dev);
    if (ret) {
        dev_err(&pdev->dev, "failed to attach reserved memory pool: %d\n", ret);
        return ret;
    }

    bcdev->miscdev.minor = MISC_DYNAMIC_MINOR;
    bcdev->miscdev.name = "biturbo-cma";
    bcdev->miscdev.fops = &biturbo_cma_fops;
    bcdev->miscdev.parent = &pdev->dev;

    ret = misc_register(&bcdev->miscdev);
    if (ret) {
        of_reserved_mem_device_release(&pdev->dev);
        return ret;
    }

    platform_set_drvdata(pdev, bcdev);
    dev_info(&pdev->dev, "biturbo CMA pool ready\n");
    return 0;
}

static int biturbo_cma_remove(struct platform_device *pdev)
{
    struct biturbo_cma_dev *bcdev = platform_get_drvdata(pdev);
    unsigned int i;

    if (!bcdev)
        return 0;

    misc_deregister(&bcdev->miscdev);

    mutex_lock(&bcdev->lock);
    for (i = 0; i < BITURBO_CMA_MAX_BUFS; i++)
        biturbo_cma_free_one(bcdev, i);
    mutex_unlock(&bcdev->lock);

    of_reserved_mem_device_release(&pdev->dev);
    return 0;
}

static const struct of_device_id biturbo_cma_of_match[] = {
    { .compatible = "biturbo,cma-pool" },
    { }
};
MODULE_DEVICE_TABLE(of, biturbo_cma_of_match);

static struct platform_driver biturbo_cma_driver = {
    .probe  = biturbo_cma_probe,
    .remove = biturbo_cma_remove,
    .driver = {
        .name = "biturbo-cma",
        .of_match_table = biturbo_cma_of_match,
    },
};

module_platform_driver(biturbo_cma_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("OpenAI Codex");
MODULE_DESCRIPTION("Biturbo CMA allocator for FPGA weight/activation buffers");
