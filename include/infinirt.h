#ifndef __INFINIRT_API_H__
#define __INFINIRT_API_H__

#include "infinicore.h"

typedef void *infinirtStream_t;
typedef void *infinirtEvent_t;

INFINI_EXTERN_C __export infiniStatus_t infinirtInit();

// Device
INFINI_EXTERN_C __export infiniStatus_t infinirtGetAllDeviceCount(int *count_array);
INFINI_EXTERN_C __export infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count);
INFINI_EXTERN_C __export infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id);
INFINI_EXTERN_C __export infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr);
INFINI_EXTERN_C __export infiniStatus_t infinirtDeviceSynchronize();

// Stream
INFINI_EXTERN_C  __export infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr);
INFINI_EXTERN_C  __export infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream);
INFINI_EXTERN_C  __export infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
INFINI_EXTERN_C  __export infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Event
typedef enum {
    INFINIRT_EVENT_COMPLETE = 0,
    INFINIRT_EVENT_NOT_READY = 1,
} infinirtEventStatus_t;

INFINI_EXTERN_C  __export infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr);
INFINI_EXTERN_C  __export infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
INFINI_EXTERN_C  __export infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr);
INFINI_EXTERN_C  __export infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event);
INFINI_EXTERN_C  __export infiniStatus_t infinirtEventDestroy(infinirtEvent_t event);

// Memory
typedef enum {
    INFINIRT_MEMCPY_H2H = 0,
    INFINIRT_MEMCPY_H2D = 1,
    INFINIRT_MEMCPY_D2H = 2,
    INFINIRT_MEMCPY_D2D = 3,
} infinirtMemcpyKind_t;

INFINI_EXTERN_C  __export infiniStatus_t infinirtMalloc(void **p_ptr, size_t size);
INFINI_EXTERN_C  __export infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size);
INFINI_EXTERN_C  __export infiniStatus_t infinirtFree(void *ptr);
INFINI_EXTERN_C  __export infiniStatus_t infinirtFreeHost(void *ptr);

INFINI_EXTERN_C  __export infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
INFINI_EXTERN_C __export infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);

// Stream-ordered memory
INFINI_EXTERN_C __export infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream);
INFINI_EXTERN_C __export infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream);

#endif // __INFINIRT_API_H__
