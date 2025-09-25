#include "rope_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rope::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}




infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {




} // namespace op::rope::cpu
