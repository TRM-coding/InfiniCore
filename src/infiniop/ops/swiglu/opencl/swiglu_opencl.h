#ifndef __SWIGLU_OPENCL_API_H__
#define __SWIGLU_OPENCL_API_H__
#include "../../../elementwise/elementwise.h"
// #include "../../operator.h"

namespace op::swiglu::opencl {
class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    op::elementwise::ElementwiseInfo _info;
    infiniDtype_t dtype;
    size_t _workspace_size;

    Descriptor(
        op::elementwise::ElementwiseInfo meta,
        infiniDtype_t dtype,
        Opaque *opaque,
        size_t workspaceSize,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          dtype(dtype),
          _opaque(opaque),
          _workspace_size(workspaceSize),
          _info(meta) {}

public:
    ~Descriptor();
    size_t workspaceSize() const { return _workspace_size; }
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};
} // namespace op::rearrange::opencl

#endif // __SWIGLU_MOORE_API_H__
