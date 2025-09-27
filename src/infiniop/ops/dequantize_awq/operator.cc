#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dequantize_awq.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/dequantize_w42f16_nvidia.cuh"
#endif

INFINI_EXTERN_C infiniStatus_t infiniopCreateDequantizeAWQDescriptor(
    infiniopHandle_t handle,
    infiniopDequantizeAWQDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t scales_desc,
    infiniopTensorDescriptor_t zeros_desc) {

#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::dequantize_awq::NAMESPACE::Descriptor::create(                     \
            handle,                                                                   \
            reinterpret_cast<op::dequantize_awq::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                                 \
            qweight_desc,                                                             \
            scales_desc,                                                              \
            zeros_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

INFINI_EXTERN_C infiniStatus_t infiniopGetDequantizeAWQWorkspaceSize(infiniopDequantizeAWQDescriptor_t desc,
                                                         size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                \
    case CASE:                                                                                              \
        *size = reinterpret_cast<const op::dequantize_awq::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

INFINI_EXTERN_C infiniStatus_t infiniopDequantizeAWQ(
    infiniopDequantizeAWQDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *qweight,
    const void *scales,
    const void *zeros,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                           \
        return reinterpret_cast<const op::dequantize_awq::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, qweight, scales, zeros, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

INFINI_EXTERN_C infiniStatus_t
infiniopDestroyDequantizeAWQDescriptor(infiniopDequantizeAWQDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        delete reinterpret_cast<const op::dequantize_awq::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}

// #endif