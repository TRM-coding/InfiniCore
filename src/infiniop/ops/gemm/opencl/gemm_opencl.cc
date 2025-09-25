#include "gemm_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include <CL/cl.h>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

namespace op::gemm::opencl {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

static const char *GemmKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef USE_FP16
typedef half data_t;
#else
typedef float data_t;
#endif

inline float load_as_float(__global const data_t *ptr, int idx) {
#ifdef USE_FP16
    return vload_half(idx, (__global const half*)ptr);
#else
    return ptr[idx];
#endif
}

inline void store_from_float(__global data_t *ptr, int idx, float value) {
#ifdef USE_FP16
    vstore_half(value, idx, (__global half*)ptr);
#else
    ptr[idx] = value;
#endif
}

__kernel void gemm_col_major(
    const int M,
    const int N,
    const int K,
    const int batch_size,
    const int lda,
    const int ldb,
    const int ldc,
    const long strideA,
    const long strideB,
    const long strideC,
    const float alpha,
    const float beta,
    __global const data_t *A,
    __global const data_t *B,
    __global data_t *C) {

    int col = get_global_id(0);   // N dimension
    int row = get_global_id(1);   // M dimension
    int batch_idx = get_global_id(2); // batch dimension
    
    if (col >= N || row >= M || batch_idx >= batch_size) return;

    // Calculate batch offsets
    long a_offset = strideA ? batch_idx * strideA : 0;
    long b_offset = strideB ? batch_idx * strideB : 0;
    long c_offset = strideC ? batch_idx * strideC : 0;

    float acc = 0.0f;
    
    // A: (M x K), B: (K x N), col-major
    for (int k = 0; k < K; ++k) {
        float a_val = load_as_float(A, a_offset + row + k * lda);
        float b_val = load_as_float(B, b_offset + k + col * ldb);
        acc += a_val * b_val;
    }
    
    float c_old = load_as_float(C, c_offset + row + col * ldc);
    float c_new = alpha * acc + beta * c_old;
    store_from_float(C, c_offset + row + col * ldc, c_new);
}
)CLC";

static std::string getBuildOptions(infiniDtype_t dtype) {
    std::string opts = "-cl-std=CL2.0";
    if (dtype == INFINI_DTYPE_F16) {
        opts += " -DUSE_FP16";
    }
    return opts;
}

static size_t getElementSize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16: return 2;
    case INFINI_DTYPE_F32: return 4;
    default: return 0;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    if (workspace_size < _workspace_size)
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;

    int M = static_cast<int>(_info.m);
    int N = static_cast<int>(_info.n);
    int K = static_cast<int>(_info.k);
    size_t batch = _info.batch;

    int lda = static_cast<int>(_info.a_matrix.ld());
    int ldb = static_cast<int>(_info.b_matrix.ld());
    int ldc = static_cast<int>(_info.c_matrix.ld());

    ptrdiff_t strideA = _info.a_matrix.stride;
    ptrdiff_t strideB = _info.b_matrix.stride;
    ptrdiff_t strideC = _info.c_matrix.stride;

    size_t elem_size = getElementSize(_dtype);

    void *device;
    void *context;
    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));
    if (!stream)
        CHECK_STATUS(infinirtGetOpenclStream(&stream));

    cl_device_id cldevice = reinterpret_cast<cl_device_id>(device);
    cl_context clcontext = reinterpret_cast<cl_context>(context);
    cl_command_queue clqueue = reinterpret_cast<cl_command_queue>(stream);

    const char *src_ptr = GemmKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr = CL_SUCCESS;
    
    cl_program program = clCreateProgramWithSource(clcontext, 1, &src_ptr, &src_len, &clerr);
    if (clerr != CL_SUCCESS || !program)
        return INFINI_STATUS_INTERNAL_ERROR;

    std::string build_opts = getBuildOptions(_dtype);
    clerr = clBuildProgram(program, 1, &cldevice, build_opts.c_str(), nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, cldevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1);
            clGetProgramBuildInfo(program, cldevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            log[log_size] = '\0';
            std::cerr << "OpenCL build error:\n" << log.data() << std::endl;
        }
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    cl_kernel kernel = clCreateKernel(program, "gemm_col_major", &clerr);
    if (clerr != CL_SUCCESS || !kernel) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 3D work group: (N, M, batch)
    size_t local_work_size[3] = {16, 16, 1};
    size_t global_work_size[3] = {
        ((size_t)N + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
        ((size_t)M + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1],
        batch
    };

    int arg_idx = 0;
    clerr  = clSetKernelArg(kernel, arg_idx++, sizeof(int), &M);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &N);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &K);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &lda);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &ldb);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &ldc);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(ptrdiff_t), &strideA);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(ptrdiff_t), &strideB);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(ptrdiff_t), &strideC);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &alpha);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &beta);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, const_cast<void *>(a));
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, const_cast<void *>(b));
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, c);

    if (clerr != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // Single kernel launch for all batches!
    clerr = clEnqueueNDRangeKernel(clqueue, kernel, 3, nullptr, 
                                   global_work_size, local_work_size, 
                                   0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::opencl
