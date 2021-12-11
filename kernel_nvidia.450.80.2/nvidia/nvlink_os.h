/*******************************************************************************
    Copyright (c) 2019 NVidia Corporation

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


#ifndef _NVLINK_OS_H_
#define _NVLINK_OS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvlink_common.h"

#define NVLINK_FREE(x) nvlink_free((void *)x)

// Memory management functions
void *      NVLINK_API_CALL nvlink_malloc(NvLength);
void        NVLINK_API_CALL nvlink_free(void *);
void *      NVLINK_API_CALL nvlink_memset(void *, int, NvLength);
void *      NVLINK_API_CALL nvlink_memcpy(void *, const void *, NvLength);
int         NVLINK_API_CALL nvlink_memcmp(const void *, const void *, NvLength);
NvU32       NVLINK_API_CALL nvlink_memRd32(const volatile void *);
void        NVLINK_API_CALL nvlink_memWr32(volatile void *, NvU32);
NvU64       NVLINK_API_CALL nvlink_memRd64(const volatile void *);
void        NVLINK_API_CALL nvlink_memWr64(volatile void *, NvU64);

// String management functions
char *      NVLINK_API_CALL nvlink_strcpy(char *, const char *);
NvLength    NVLINK_API_CALL nvlink_strlen(const char *);
int         NVLINK_API_CALL nvlink_strcmp(const char *, const char *);
int         NVLINK_API_CALL nvlink_snprintf(char *, NvLength, const char *, ...);

// Locking support functions
void *      NVLINK_API_CALL nvlink_allocLock(void);
void        NVLINK_API_CALL nvlink_acquireLock(void *);
NvBool      NVLINK_API_CALL nvlink_isLockOwner(void *);
void        NVLINK_API_CALL nvlink_releaseLock(void *);
void        NVLINK_API_CALL nvlink_freeLock(void *);

// Miscellaneous functions
void        NVLINK_API_CALL nvlink_assert(int expression);
void        NVLINK_API_CALL nvlink_sleep(unsigned int ms);
void        NVLINK_API_CALL nvlink_print(const char *, int, const char *, int, const char *, ...);
int         NVLINK_API_CALL nvlink_is_admin(void);

// Capability functions
NvlStatus NVLINK_API_CALL nvlink_acquire_fabric_mgmt_cap(void *osPrivate, NvU64 capDescriptor);
int NVLINK_API_CALL nvlink_is_fabric_manager(void *osPrivate);

#define NVLINK_DBG_LEVEL_INFO       0x0
#define NVLINK_DBG_LEVEL_SETUP      0x1
#define NVLINK_DBG_LEVEL_USERERRORS 0x2
#define NVLINK_DBG_LEVEL_WARNINGS   0x3
#define NVLINK_DBG_LEVEL_ERRORS     0x4

#define NVLINK_DBG_WHERE       __FILE__, __LINE__, __FUNCTION__
#define NVLINK_DBG_INFO        NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_INFO
#define NVLINK_DBG_SETUP       NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_SETUP
#define NVLINK_DBG_USERERRORS  NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_USERERRORS
#define NVLINK_DBG_WARNINGS    NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_WARNINGS
#define NVLINK_DBG_ERRORS      NVLINK_DBG_WHERE, NVLINK_DBG_LEVEL_ERRORS

#ifdef __cplusplus
}
#endif

#endif //_NVLINK_OS_H_
