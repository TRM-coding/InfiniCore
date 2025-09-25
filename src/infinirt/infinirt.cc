#include "infinirt.h"
#include "../utils.h"
#include "ascend/infinirt_ascend.h"
#include "bang/infinirt_bang.h"
#include "cpu/infinirt_cpu.h"
#include "cuda/infinirt_cuda.cuh"
#include "kunlun/infinirt_kunlun.h"
#include "metax/infinirt_metax.h"
#include "moore/infinirt_moore.h"
#include "opencl/infinirt_opencl.h"

thread_local infiniDevice_t CURRENT_DEVICE_TYPE = INFINI_DEVICE_CPU;
thread_local int CURRENT_DEVICE_ID = 0;

INFINI_EXTERN_C infiniStatus_t infinirtInit() {
#if defined(ENABLE_ASCEND_API)
    CHECK_STATUS(infinirt::ascend::init());
#elif defined(ENABLE_OPENCL_API)
    CHECK_STATUS(infinirt::opencl::init());
#endif
    return INFINI_STATUS_SUCCESS;
}

INFINI_EXTERN_C infiniStatus_t infinirtGetAllDeviceCount(int *count_array) {
    if (count_array == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    for (size_t i = 0; i < INFINI_DEVICE_TYPE_COUNT; i++) {
        if (i == INFINI_DEVICE_ILUVATAR || i == INFINI_DEVICE_KUNLUN || i == INFINI_DEVICE_SUGON) {
            count_array[i] = 0;
            continue;
        }
        auto status = infinirtGetDeviceCount(static_cast<infiniDevice_t>(i), &count_array[i]);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
    }
    return INFINI_STATUS_SUCCESS;
}

INFINI_EXTERN_C infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr) {
    if (device_ptr == nullptr && device_id_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (device_ptr != nullptr) {
        *device_ptr = CURRENT_DEVICE_TYPE;
    }
    if (device_id_ptr != nullptr) {
        *device_id_ptr = CURRENT_DEVICE_ID;
    }

    return INFINI_STATUS_SUCCESS;
}

#define INFINIRT_CALL_DEVICE_API_AND(DEVICE_TYPE, API, PARAMS, ACTION) \
    {                                                                  \
        infiniStatus_t _status;                                        \
        switch (DEVICE_TYPE) {                                         \
        case INFINI_DEVICE_CPU:                                        \
            _status = infinirt::cpu::API PARAMS;                       \
            break;                                                     \
        case INFINI_DEVICE_NVIDIA:                                     \
            _status = infinirt::cuda::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_CAMBRICON:                                  \
            _status = infinirt::bang::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_ASCEND:                                     \
            _status = infinirt::ascend::API PARAMS;                    \
            break;                                                     \
        case INFINI_DEVICE_METAX:                                      \
            _status = infinirt::metax::API PARAMS;                     \
            break;                                                     \
        case INFINI_DEVICE_MOORE:                                      \
            _status = infinirt::musa::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_KUNLUN:                                     \
            _status = infinirt::kunlun::API PARAMS;                    \
            break;                                                     \
        case INFINI_DEVICE_ILUVATAR:                                   \
            _status = infinirt::cuda::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_OPENCL:                                     \
            _status = infinirt::opencl::API PARAMS;                    \
            break;                                                     \
        default:                                                       \
            _status = INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;         \
        }                                                              \
        { ACTION; }                                                    \
        return _status;                                                \
    }

#define INFINIRT_CALL_DEVICE_API(API, PARAMS) INFINIRT_CALL_DEVICE_API_AND(CURRENT_DEVICE_TYPE, API, PARAMS, )

INFINI_EXTERN_C infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count) {
    if (count == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    INFINIRT_CALL_DEVICE_API_AND(device, getDeviceCount, (count), {});
}

INFINI_EXTERN_C infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id) {
    INFINIRT_CALL_DEVICE_API_AND(device, setDevice, (device_id),
                                 { CURRENT_DEVICE_TYPE = device;
                                   CURRENT_DEVICE_ID = device_id; });
}

INFINI_EXTERN_C infiniStatus_t infinirtDeviceSynchronize() {
    INFINIRT_CALL_DEVICE_API(deviceSynchronize, ());
}

INFINI_EXTERN_C infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr) {
    INFINIRT_CALL_DEVICE_API(streamCreate, (stream_ptr));
}

INFINI_EXTERN_C infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(streamDestroy, (stream));
}

INFINI_EXTERN_C infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(streamSynchronize, (stream));
}

INFINI_EXTERN_C infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    INFINIRT_CALL_DEVICE_API(streamWaitEvent, (stream, event));
}

INFINI_EXTERN_C infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr) {
    INFINIRT_CALL_DEVICE_API(eventCreate, (event_ptr));
}

INFINI_EXTERN_C infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(eventRecord, (event, stream));
}

INFINI_EXTERN_C infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    INFINIRT_CALL_DEVICE_API(eventQuery, (event, status_ptr));
}

INFINI_EXTERN_C infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event) {
    INFINIRT_CALL_DEVICE_API(eventSynchronize, (event));
}

INFINI_EXTERN_C infiniStatus_t infinirtEventDestroy(infinirtEvent_t event) {
    INFINIRT_CALL_DEVICE_API(eventDestroy, (event));
}

INFINI_EXTERN_C infiniStatus_t infinirtMalloc(void **p_ptr, size_t size) {
    INFINIRT_CALL_DEVICE_API(mallocDevice, (p_ptr, size));
}

INFINI_EXTERN_C infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size) {
    INFINIRT_CALL_DEVICE_API(mallocHost, (p_ptr, size));
}

INFINI_EXTERN_C infiniStatus_t infinirtFree(void *ptr) {
    INFINIRT_CALL_DEVICE_API(freeDevice, (ptr));
}

INFINI_EXTERN_C infiniStatus_t infinirtFreeHost(void *ptr) {
    INFINIRT_CALL_DEVICE_API(freeHost, (ptr));
}

INFINI_EXTERN_C infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    INFINIRT_CALL_DEVICE_API(memcpy, (dst, src, size, kind));
}

INFINI_EXTERN_C infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(memcpyAsync, (dst, src, size, kind, stream));
}

INFINI_EXTERN_C infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(mallocAsync, (p_ptr, size, stream));
}

INFINI_EXTERN_C infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(freeAsync, (ptr, stream));
}
