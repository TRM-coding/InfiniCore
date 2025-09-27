#include "swiglu_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../../../tensor.h"
#include "infiniop/handle.h"
#include "infinirt.h"
#include <CL/cl.h>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

static const char *SwigluKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef REAL_T
#define REAL_T float
#endif

typedef long stride_t;

#ifdef USE_HALF
inline float real_to_float(half v) { return convert_float(v); }
inline half float_to_real(float v) { return convert_half(v); }
#else
inline float real_to_float(REAL_T v) { return (float)v; }
inline REAL_T float_to_real(float v) { return (REAL_T)v; }
#endif

kernel void swiglu_kernel(
    global REAL_T *y,
    int ndim,
    global const size_t *output_shape,
    global const stride_t *output_strides,

    global const REAL_T *a,
    global const size_t *a_shape,
    global const stride_t *a_strides,

    global const REAL_T *b,
    global const size_t *b_shape,
    global const stride_t *b_strides,

    int total_size
) {
    int gid = get_global_id(0);
    if (gid >= total_size) {
        return;
    }

    size_t remaining = (size_t)gid;
    long out_offset = 0;
    long a_offset = 0;
    long b_offset = 0;

    for (int d = ndim - 1; d >= 0; --d) {
        size_t dim = output_shape[d];
        size_t idx = dim == 0 ? 0 : remaining % dim;
        remaining = dim == 0 ? 0 : remaining / dim;

        out_offset += (long)(idx) * output_strides[d];
        a_offset += ((a_shape[d] == 1) ? 0 : (long)(idx)) * a_strides[d];
        b_offset += ((b_shape[d] == 1) ? 0 : (long)(idx)) * b_strides[d];
    }

    float gate = real_to_float(b[b_offset]);
    float up = real_to_float(a[a_offset]);
    float sig = 1.0f / (1.0f + exp(-gate));
    y[out_offset] = float_to_real(up * gate * sig);
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
    // 不支持 BF16
    case INFINI_DTYPE_BF16:
        return false;
    default:
        return false;
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

static size_t tensorElementCount(const size_t *shape, int ndim) {
    size_t elems = 1;
    for (int i = 0; i < ndim; ++i) {
        size_t dim = shape[i];
        elems *= dim == 0 ? 1 : dim;
    }
    return elems;
}

static size_t tensorStorageElementCount(const size_t *shape, const ptrdiff_t *strides, int ndim) {
    if (ndim == 0) {
        return 1;
    }
    ptrdiff_t min_offset = 0;
    ptrdiff_t max_offset = 0;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            return 0;
        }
        ptrdiff_t extent = strides[i] * static_cast<ptrdiff_t>(shape[i] - 1);
        if (extent > 0) {
            max_offset += extent;
        } else {
            min_offset += extent;
        }
    }
    return static_cast<size_t>(max_offset - min_offset + 1);
}

namespace op::swiglu::opencl {

Descriptor::~Descriptor() = default;
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &up_desc = input_desc_vec.at(0);
    const auto &gate_desc = input_desc_vec.at(1);
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    *desc_ptr = new Descriptor(
        info_result.take(),
        dtype,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    op::elementwise::ElementwiseInfo _info,
    infiniDtype_t dtype,
    void *output,
    std::vector<const void *> inputs,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {
    auto ndim = _info.getNdim();
    auto outputsize = _info.getOutputSize();
    auto inputsize = _info.getInputSize();
    auto input_a_matrix = inputs[0];
    auto input_a_matrix_stride = _info.getInputStrides(0);
    auto input_a_shape = _info.getInputShape(0);
    auto input_b_matrix = inputs[1];
    auto input_b_shape = _info.getInputShape(1);
    auto input_b_matrix_stride = _info.getInputStrides(1);
    auto output_stride = _info.getOutputStrides();
    auto output_shape = _info.getOutputShape();
    size_t dtype_bytes = dtypeSize(dtype);
    if (!dtype_bytes) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    size_t output_storage_bytes = tensorStorageElementCount(output_shape, output_stride, ndim) * dtype_bytes;
    size_t input_a_storage_bytes = tensorStorageElementCount(input_a_shape, input_a_matrix_stride, ndim) * dtype_bytes;
    size_t input_b_storage_bytes = tensorStorageElementCount(input_b_shape, input_b_matrix_stride, ndim) * dtype_bytes;

    // 创建程序对象
    const char *src_ptr = SwigluKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);

    std::string cl_type;
    if (!dtypeToClType(dtype, cl_type)) {
        clReleaseProgram(program);
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    std::string build_opts;
    build_opts += "-cl-std=CL2.0 ";
    build_opts += "-DREAL_T=" + cl_type + " ";
    if (dtype == INFINI_DTYPE_F16) {
        build_opts += "-DUSE_HALF ";
    }
    clerr = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);

    // 获取内核代码
    cl_kernel kernel = clCreateKernel(program, "swiglu_kernel", &clerr);
    int arg_idx = 0;

    // y 参数
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, output);
    if (clerr != CL_SUCCESS) {
        if (output_storage_bytes) {
            infinirtMalloc(&y_svm, output_storage_bytes);
            infinirtMemcpy(y_svm, output, output_storage_bytes, INFINIRT_MEMCPY_H2D);
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }

    cl_int cl_ndim = static_cast<cl_int>(ndim);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_ndim);

    // output_shape
    void *output_shape_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, (void *)output_shape);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim * sizeof(size_t);
        infinirtMalloc(&output_shape_svm, num_bytes);
        infinirtMemcpy(output_shape_svm, output_shape, num_bytes, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, output_shape_svm);
    }
    // output_strides
    void *output_strides_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, (void *)output_stride);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim * sizeof(ptrdiff_t);
        infinirtMalloc(&output_strides_svm, num_bytes);
        infinirtMemcpy(output_strides_svm, output_stride, num_bytes, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, output_strides_svm);
    }

    // a matrix (up)
    void *a_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, const_cast<void *>(input_a_matrix));
    if (clerr != CL_SUCCESS) {
        if (input_a_storage_bytes) {
            infinirtMalloc(&a_svm, input_a_storage_bytes);
            infinirtMemcpy(a_svm, input_a_matrix, input_a_storage_bytes, INFINIRT_MEMCPY_H2D);
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a_svm);
    }

    // a_shape
    void *a_shape_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, (void *)input_a_shape);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim * sizeof(size_t);
        infinirtMalloc(&a_shape_svm, num_bytes);
        infinirtMemcpy(a_shape_svm, input_a_shape, num_bytes, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a_shape_svm);
    }
    // a_strides
    void *a_stride_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, (void *)input_a_matrix_stride);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim * sizeof(ptrdiff_t);
        infinirtMalloc(&a_stride_svm, num_bytes);
        infinirtMemcpy(a_stride_svm, input_a_matrix_stride, num_bytes, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a_stride_svm);
    }

    // b matrix (gate)
    void *b_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, const_cast<void *>(input_b_matrix));
    if (clerr != CL_SUCCESS) {
        if (input_b_storage_bytes) {
            infinirtMalloc(&b_svm, input_b_storage_bytes);
            infinirtMemcpy(b_svm, input_b_matrix, input_b_storage_bytes, INFINIRT_MEMCPY_H2D);
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b_svm);
    }

    // b_shape
    void *b_shape_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, (void *)input_b_shape);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim * sizeof(size_t);
        infinirtMalloc(&b_shape_svm, num_bytes);
        infinirtMemcpy(b_shape_svm, input_b_shape, num_bytes, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b_shape_svm);
    }
    // b_strides
    void *b_stride_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, (void *)input_b_matrix_stride);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim * sizeof(ptrdiff_t);
        infinirtMalloc(&b_stride_svm, num_bytes);
        infinirtMemcpy(b_stride_svm, input_b_matrix_stride, num_bytes, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b_stride_svm);
    }

    cl_int cl_total_size = static_cast<cl_int>(outputsize);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_total_size);

    size_t global_work_size[1] = {outputsize};

    // 启动 OpenCL kernel
    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    clFinish(cl_queue);

    // 拷贝回输出
    if (y_svm && output_storage_bytes) {
        infinirtMemcpy(output, y_svm, output_storage_bytes, INFINIRT_MEMCPY_D2H);
    }

    // 释放内存
    if (y_svm) { infinirtFree(y_svm); }
    if (a_svm) { infinirtFree(a_svm); }
    if (b_svm) { infinirtFree(b_svm); }
    if (output_shape_svm) { infinirtFree(output_shape_svm); }
    if (output_strides_svm) { infinirtFree(output_strides_svm); }
    if (a_shape_svm) { infinirtFree(a_shape_svm); }
    if (a_stride_svm) { infinirtFree(a_stride_svm); }
    if (b_shape_svm) { infinirtFree(b_shape_svm); }
    if (b_stride_svm) { infinirtFree(b_stride_svm); }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    void *device;
    void *context;

    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));

    auto device_cl = reinterpret_cast<cl_device_id>(device);
    auto context_cl = reinterpret_cast<cl_context>(context);

    // 获取context中的设别数量
    cl_uint num_devices;
    auto err_c = clGetContextInfo(context_cl, CL_CONTEXT_NUM_DEVICES, sizeof(num_devices), &num_devices, nullptr);

    // 获取context中的设别列表
    cl_device_id *devices_in_context = new cl_device_id[num_devices];
    err_c = clGetContextInfo(context_cl, CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), devices_in_context, nullptr);

    auto clcontext = static_cast<cl_context>(context);
    auto cldevice = static_cast<cl_device_id>(device);

    if (!stream) {
        CHECK_STATUS(infinirtGetOpenclStream(&stream));
    }
    auto clqueue = static_cast<cl_command_queue>(stream);
    CHECK_STATUS(launchKernel(_info, dtype, output, inputs, clcontext, cldevice, clqueue));

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::opencl