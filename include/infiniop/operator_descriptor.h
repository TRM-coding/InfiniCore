#ifndef __INFINIOP_OPERATOR_DESCRIPTOR_API_H__
#define __INFINIOP_OPERATOR_DESCRIPTOR_API_H__

#include "handle.h"
#include "tensor_descriptor.h"

// Base descriptor for all operators
struct InfiniopDescriptor;

INFINI_EXTERN_C __export infiniStatus_t infiniopGetDescriptorDeviceType(const struct InfiniopDescriptor *desc_ptr, infiniDevice_t *device_type);
INFINI_EXTERN_C __export infiniStatus_t infiniopGetDescriptorDeviceId(const struct InfiniopDescriptor *desc_ptr, int *device_id);

#endif //__INFINIOP_OPERATOR_DESCRIPTOR_API_H__
