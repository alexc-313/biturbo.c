#ifndef BITURBO_CMA_IOCTL_H
#define BITURBO_CMA_IOCTL_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
typedef __u32 bt_cma_u32;
typedef __u64 bt_cma_u64;
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint32_t bt_cma_u32;
typedef uint64_t bt_cma_u64;
#endif

#define BITURBO_CMA_DEVICE "/dev/biturbo-cma"
#define BITURBO_CMA_NAME_LEN 32
#define BITURBO_CMA_IOCTL_MAGIC 'B'

typedef struct {
    bt_cma_u64 size;
    bt_cma_u64 dma_addr;
    bt_cma_u64 mmap_offset;
    bt_cma_u32 handle;
    bt_cma_u32 flags;
    char       name[BITURBO_CMA_NAME_LEN];
} biturbo_cma_alloc_req_t;

typedef struct {
    bt_cma_u32 handle;
    bt_cma_u32 reserved;
} biturbo_cma_free_req_t;

#define BITURBO_CMA_IOCTL_ALLOC \
    _IOWR(BITURBO_CMA_IOCTL_MAGIC, 0x01, biturbo_cma_alloc_req_t)
#define BITURBO_CMA_IOCTL_FREE \
    _IOW(BITURBO_CMA_IOCTL_MAGIC, 0x02, biturbo_cma_free_req_t)

#endif /* BITURBO_CMA_IOCTL_H */
