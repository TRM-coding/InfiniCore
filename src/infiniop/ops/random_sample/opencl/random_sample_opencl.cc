#include "random_sample_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../info.h"
#include "infinicore.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include <CL/cl.h>
#include <cstring>
#include <algorithm>

static const char *RandomSampleKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

#ifndef Tidx
#define Tidx int
#endif

#ifndef Tcompute
#define Tcompute float
#endif

typedef struct {
    Tidx idx;
    Tcompute val;
} KVPair;

kernel void random_sample_argmax(
    global Tidx *result,
    global Tval const *probs,
    int n) {
    
    if (get_global_id(0) != 0) return;
    
    Tidx max_idx = 0;
    Tcompute max_val = (Tcompute)probs[0];
    
    for (int i = 1; i < n; i++) {
        Tcompute val = (Tcompute)probs[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    result[0] = max_idx;
}

kernel void random_sample_random(
    global Tidx *result,
    global Tval const *probs,
    global KVPair *workspace_pairs,
    int n,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    
    if (get_global_id(0) != 0) return;
    
    // Build pairs
    for (int i = 0; i < n; i++) {
        workspace_pairs[i].idx = i;
        workspace_pairs[i].val = (Tcompute)probs[i];
    }
    
    // Simple bubble sort (for small n)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (workspace_pairs[j].val < workspace_pairs[j + 1].val) {
                KVPair temp = workspace_pairs[j];
                workspace_pairs[j] = workspace_pairs[j + 1];
                workspace_pairs[j + 1] = temp;
            }
        }
    }
    
    // Softmax with temperature
    Tcompute max_val = workspace_pairs[0].val;
    workspace_pairs[0].val = 1.0f;
    for (int i = 1; i < n; i++) {
        workspace_pairs[i].val = workspace_pairs[i - 1].val + 
            exp((workspace_pairs[i].val - max_val) / temperature);
    }
    
    // Calculate limits
    int topk_limit = min(topk, n);
    Tcompute pk = workspace_pairs[topk_limit - 1].val;
    Tcompute pp = workspace_pairs[n - 1].val * topp;
    Tcompute plimit = random_val * min(pk, pp);
    
    // Sample
    result[0] = workspace_pairs[n - 1].idx; // default to last
    for (int i = 0; i < n; i++) {
        if (plimit <= workspace_pairs[i].val) {
            result[0] = workspace_pairs[i].idx;
            break;
        }
    }
}
)CLC";

inline size_t dtypeSize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_BYTE:
        return 1;
    case INFINI_DTYPE_BOOL:
        return 1;
    case INFINI_DTYPE_I8:
        return 1;
    case INFINI_DTYPE_U8:
        return 1;
    case INFINI_DTYPE_I16:
        return 2;
    case INFINI_DTYPE_U16:
        return 2;
    case INFINI_DTYPE_F16:
        return 2;
    case INFINI_DTYPE_I32:
        return 4;
    case INFINI_DTYPE_U32:
        return 4;
    case INFINI_DTYPE_F32:
        return 4;
    case INFINI_DTYPE_I64:
        return 8;
    case INFINI_DTYPE_U64:
        return 8;
    case INFINI_DTYPE_F64:
        return 8;
    default:
        return 0;
    }
}

static bool dtypeToClType(infiniDtype_t dt, std::string &out) {
    switch (dt) {
    case INFINI_DTYPE_F32:
        out = "float";
        return true;
    case INFINI_DTYPE_F16:
        out = "half";
        return true;
    case INFINI_DTYPE_I32:
        out = "int";
        return true;
    default:
        return false;
    }
}

static const char *clErrorString(cl_int err) {
    switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    default: return "UNKNOWN_CL_ERROR";
    }
}

namespace op::random_sample::opencl {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);

    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    
    // Calculate workspace size for KVPair array when doing random sampling
    size_t workspace_size = info.n * (sizeof(int) + sizeof(float)); // KVPair struct size

    *desc_ptr = new Descriptor(
        std::move(info),
        workspace_size,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchRandomSampleKernel(
    void *result, infiniDtype_t result_dtype,
    const void *probs, infiniDtype_t probs_dtype,
    size_t n, float random_val, float topp, int topk, float temperature,
    void *workspace, size_t workspace_size,
    cl_context context, cl_device_id device, cl_command_queue cl_queue) {
    
    std::string dt_val, dt_idx, dt_compute;
    dt_compute = "float";
    
    if (!dtypeToClType(probs_dtype, dt_val)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtypeToClType(result_dtype, dt_idx)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    const char *src_ptr = RandomSampleKernelSource;
    size_t src_len = std::strlen(src_ptr);

    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);
    if (clerr != CL_SUCCESS || program == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    std::string build_opts;
    build_opts += "-D Tval=" + dt_val + " ";
    build_opts += "-D Tidx=" + dt_idx + " ";
    build_opts += "-D Tcompute=" + dt_compute + " ";
    build_opts += "-cl-std=CL2.0 ";

    clerr = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // Choose kernel based on random_val
    const char *kernel_name = (random_val < 0) ? "random_sample_argmax" : "random_sample_random";
    cl_kernel kernel = clCreateKernel(program, kernel_name, &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // Set kernel arguments
    int arg_idx = 0;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, result);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, probs);
    
    if (random_val >= 0) {
        // For random sampling, need workspace for pairs
        clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, workspace);
    }
    
    cl_int n_cl = static_cast<cl_int>(n);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &n_cl);
    
    if (random_val >= 0) {
        clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &random_val);
        clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &topp);
        clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &topk);
        clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &temperature);
    }

    if (clerr != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    size_t global_size = 1;
    size_t local_size = 1;
    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    
    if (clerr != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {

    // if (workspace_size < _workspace_size) {
    //     return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    // }

    size_t n = _info.n;  // Use correct field name from RandomSampleInfo
    
    void *device;
    void *context;
    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));
    
    if (!stream) {
        CHECK_STATUS(infinirtGetOpenclStream(&stream));
    }
    
    cl_context clcontext = static_cast<cl_context>(context);
    cl_device_id cl_device = static_cast<cl_device_id>(device);
    cl_command_queue cl_queue = static_cast<cl_command_queue>(stream);
    
    CHECK_STATUS(launchRandomSampleKernel(
        result, _info.dt_i,      // Use correct field name for result dtype
        probs, _info.dt_p,       // Use correct field name for probs dtype
        n, random_val, topp, topk, temperature,
        workspace, workspace_size,
        clcontext, cl_device, cl_queue));
        
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::opencl
