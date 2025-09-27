#ifndef __REARRANGE_OPENCL_H__
#define __REARRANGE_OPENCL_H__

#include "../rearrange.h"

namespace op::rearrange::opencl {
class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    utils::RearrangeMeta _meta;
    infiniDtype_t dtype;

    Descriptor(
        utils::RearrangeMeta meta,
        infiniDtype_t dtype,
        Opaque *opaque,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
            dtype(dtype),
          _opaque(opaque),
          _meta(meta) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc);

    infiniStatus_t calculate(
        void *y,
        const void *x,
        void *stream) const;
};
} // namespace op::rearrange::NAMESPACE

#endif // __REARRANGE_CPU_H__
