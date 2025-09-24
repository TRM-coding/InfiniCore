#include "mul_opencl.h"

#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include <CL/cl.h>
#include <fstream>
#include <memory>
#include <sstream>

static const char *MulKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Ta
#define Ta float
#endif

#ifndef Tb
#define Tb float
#endif

#ifndef Tc
#define Tc float
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 1
#endif

typedef unsigned int Tidx;

kernel void mul(
    global Tc *y_,
    int const s_y_batch,
    global Ta const *a_,
    int const s_a_batch,
    global Tb const *b_,
    int const s_b_batch,
    Tidx const batch_size,
    Tidx const d) {

    Tidx g_idx = get_group_id(0),
         l_idx = get_local_id(0),
         l_len = get_local_size(0);
    
    if (g_idx >= batch_size) return;
    
    global Tc *y = y_ + g_idx * s_y_batch;
    global Ta const *a = a_ + g_idx * s_a_batch;
    global Tb const *b = b_ + g_idx * s_b_batch;

    for (Tidx idx = l_idx; idx < d; idx += l_len) {
        y[idx] = (Tc)((Tc)a[idx] * (Tc)b[idx]);
    }
}
)CLC";

namespace op::mul::opencl {

struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    std::vector<infiniopTensorDescriptor_t> inputs) {
    auto result = MulInfo::create(c_desc, inputs[0], inputs[1]);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto a_tensor = inputs[0];
    auto b_tensor = inputs[1];
    auto a_strides = _info.x_strides[0];
    auto b_strides = _info.x_strides[1];
    auto y_strides = _info.y_strides;

    auto dim = _info.dim();
    uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);
    size_t block_size = _opaque->internal->maxThreadsPerBlock();
    void *device;
    void *context;

    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));
    cl_context clcontext = static_cast<cl_context>(context);
    cl_device_id cldevice = static_cast<cl_device_id>(device);
    if (!stream) {
        CHECK_STATUS(infinirtGetOpenclStream(&stream));
    }
    cl_command_queue clqueue = static_cast<cl_command_queue>(stream);
    
    CHECK_STATUS(launchKernel(batch_size, dim, 
                             output, _info.atype, y_strides[0],
                             a_tensor, _info.atype, a_strides[0], 
                             b_tensor, _info.btype, b_strides[0],
                             block_size, clcontext, cldevice, clqueue));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::mul::opencl

static bool dtypeToClType(infiniDtype_t dt, std::string &out) {
    switch (dt) {
    case INFINI_DTYPE_F32:
        out = "float";
        return true;
    case INFINI_DTYPE_F16:
        out = "half";
        return true;
    // 不支持 BF16
    case INFINI_DTYPE_BF16:
        return false;
    default:
        return false;
    }
}

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

// debug todo:移动到common
static const char *clErrorString(cl_int err) {
    switch (err) {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    default:
        return "UNKNOWN_CL_ERROR";
    }
}

// launch kernel
infiniStatus_t launchKernel(
    uint32_t batch_size, size_t dim,
    void *y, infiniDtype_t ytype, ptrdiff_t stride_y_batch,
    const void *a, infiniDtype_t atype, ptrdiff_t stride_a_batch,
    const void *b, infiniDtype_t btype, ptrdiff_t stride_b_batch,
    size_t block_size,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {
    
    std::string dt_a, dt_b, dt_compute;
    dt_compute = "float";
    if (!dtypeToClType(atype, dt_a)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtypeToClType(btype, dt_b)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    size_t items_perthread = (dim + block_size - 1) / block_size;

    const char *src_ptr = MulKernelSource;
    size_t src_len = std::strlen(src_ptr);

    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);
    if (clerr != CL_SUCCESS || program == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Ta=" + dt_a + " ";
    build_opts += "-D Tb=" + dt_b + " ";
    build_opts += "-D Tc=" + dt_compute + " ";
    build_opts += "-D ITEMS_THREAD=" + std::to_string(items_perthread) + " ";
    build_opts += "-cl-std=CL2.0 ";

    clerr = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        // build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            log[log_size] = '\0';
            printf("OpenCL build log: %s\n", log.data());
        }
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    cl_kernel kernel = clCreateKernel(program, "mul", &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&y_svm, ((batch_size - 1) * stride_y_batch + dim) * dtypeSize(ytype));
        infinirtMemcpy(y_svm, y, ((batch_size - 1) * stride_y_batch + dim) * dtypeSize(ytype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }
    cl_int s_y_batch = static_cast<cl_int>(stride_y_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_batch);
    
    void *a_svm = NULL;
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, a);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&a_svm, ((batch_size - 1) * stride_a_batch + dim) * dtypeSize(atype));
        infinirtMemcpy(a_svm, a, ((batch_size - 1) * stride_a_batch + dim) * dtypeSize(atype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a_svm);
    }
    cl_int s_a_batch = static_cast<cl_int>(stride_a_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_a_batch);
    
    void *b_svm = NULL;
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, b);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&b_svm, ((batch_size - 1) * stride_b_batch + dim) * dtypeSize(btype));
        infinirtMemcpy(b_svm, b, ((batch_size - 1) * stride_b_batch + dim) * dtypeSize(btype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b_svm);
    }
    cl_int s_b_batch = static_cast<cl_int>(stride_b_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_b_batch);
    
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &batch_size);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &dim);

    size_t global_size = batch_size * block_size;

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &global_size, &block_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size, block_size);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    if (y_svm) { // for python test
        infinirtMemcpy(y, y_svm, ((batch_size - 1) * stride_y_batch + dim) * dtypeSize(ytype), INFINIRT_MEMCPY_D2H);
    }

    // cleanup program/kernel
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return INFINI_STATUS_SUCCESS;
}