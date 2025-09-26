#include "causal_softmax_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "infiniop/handle.h"
#include "infinirt.h"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

static const char *CausalSoftmaxKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef SCALAR_T
#define SCALAR_T float
#endif

#ifndef COMPUTE_T
#define COMPUTE_T float
#endif

kernel void causal_softmax_kernel(
    global SCALAR_T* x,
    int const x_stride_batch,
    int const x_stride_i,
    int const x_stride_j,
    global SCALAR_T* y,
    int const y_stride_batch,
    int const y_stride_i,
    int const y_stride_j,
    int const seq_len,
    int const total_seq_len
){
    size_t i = get_global_id(0);      // sequence index within current window
    size_t b = get_global_id(1);      // batch index

    if (i >= (size_t)seq_len) return;

    int max_j = (total_seq_len - seq_len) + (int)i;

    size_t x_base = (size_t)b * (size_t)x_stride_batch + (size_t)i * (size_t)x_stride_i;
    size_t y_base = (size_t)b * (size_t)y_stride_batch + (size_t)i * (size_t)y_stride_i;

    // Mask future positions to zero
    for (int j = max_j + 1; j < total_seq_len; ++j) {
        size_t y_off = y_base + (size_t)j * (size_t)y_stride_j;
        y[y_off] = (SCALAR_T)(0.0f);
    }

    // Find max for numerical stability
    COMPUTE_T max_val = -INFINITY;
    for (int j = 0; j <= max_j; ++j) {
        size_t x_off = x_base + (size_t)j * (size_t)x_stride_j;
        COMPUTE_T v = (COMPUTE_T)(x[x_off]);
        if (v > max_val) max_val = v;
    }

    // Exponentiate and accumulate sum
    COMPUTE_T sum = 0.0f;
    for (int j = 0; j <= max_j; ++j) {
        size_t x_off = x_base + (size_t)j * (size_t)x_stride_j;
        size_t y_off = y_base + (size_t)j * (size_t)y_stride_j;
        COMPUTE_T e = exp((COMPUTE_T)(x[x_off]) - max_val);
        sum += e;
        y[y_off] = (SCALAR_T)(e);
    }

    // Normalize
    COMPUTE_T inv_sum = 1.0f / sum;
    for (int j = 0; j <= max_j; ++j) {
        size_t y_off = y_base + (size_t)j * (size_t)y_stride_j;
        y[y_off] = (SCALAR_T)(((COMPUTE_T)(y[y_off])) * inv_sum);
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
    // 不支持 BF16
    case INFINI_DTYPE_BF16:
        return false;
    default:
        return false;
    }
}

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

namespace op::causal_softmax::opencl {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto result = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    const CausalSoftmaxInfo &info,
    void *y,const void *x, cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {
    
    // 获取算子元数据
    auto dtype=info.dtype;
    std::string dt;
    std::string dt_compute = "float";
    if (!dtypeToClType(dtype, dt)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto batch_size = info.batch_size;
    auto seq_number = info.seq_len;
    auto total_seq_len = info.total_seq_len;
    auto y_stride_batch = info.y_stride_b;
    auto y_stride_i = info.y_stride_i;
    auto y_stride_j = info.y_stride_j;
    auto x_stride_batch = info.x_stride_b;
    auto x_stride_i = info.x_stride_i;
    auto x_stride_j = info.x_stride_j;
    
    
    // 创建程序对象
    const char * src_ptr = CausalSoftmaxKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context,1,&src_ptr,&src_len,&clerr);

    // 构造编译命令并完成编译
    std::string build_opts;
    build_opts += "-cl-std=CL2.0 ";
    build_opts += ("-D SCALAR_T=" + dt + " ");
    build_opts += ("-D COMPUTE_T=" + dt_compute + " ");
    clerr=clBuildProgram(program,1,&device,build_opts.c_str(),nullptr,nullptr);

    // 获取内核代码
    cl_kernel kernel = clCreateKernel(program,"causal_softmax_kernel",&clerr); 
    int arg_idx=0;

    // X矩阵参数传入
    void *x_svm=NULL;
    clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,const_cast<void*>(x));
    if(clerr != CL_SUCCESS)
    {
        size_t num_elems =
            (batch_size - 1) * x_stride_batch +
            (seq_number - 1) * x_stride_i +
            (total_seq_len - 1) * x_stride_j + 1;
        infinirtMalloc(&x_svm,num_elems*dtypeSize(dtype));
        infinirtMemcpy(x_svm,x,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,x_svm);
    }
    cl_int cl_x_stride_batch = static_cast<cl_int>(x_stride_batch);
    cl_int cl_x_stride_i=static_cast<cl_int>(x_stride_i);
    cl_int cl_x_stride_j=static_cast<cl_int>(x_stride_j);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_x_stride_batch);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_x_stride_i);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_x_stride_j);

    // Y矩阵参数传入
    void *y_svm=NULL;
    clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,y);
    if(clerr != CL_SUCCESS)
    {
        size_t num_elems =
            (batch_size - 1) * y_stride_batch +
            (seq_number - 1) * y_stride_i +
            (total_seq_len - 1) * y_stride_j + 1;
        infinirtMalloc(&y_svm,num_elems*dtypeSize(dtype));
        infinirtMemcpy(y_svm,y,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,y_svm);
    }
    cl_int cl_y_stride_batch = static_cast<cl_int>(y_stride_batch);
    cl_int cl_y_stride_i=static_cast<cl_int>(y_stride_i); // fix: was y_stride_batch
    cl_int cl_y_stride_j=static_cast<cl_int>(y_stride_j);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_y_stride_batch);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_y_stride_i);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_y_stride_j);

    // 传入参数 seq_number, total_seq_len
    cl_int cl_seq_number=static_cast<cl_int>(seq_number);
    cl_int cl_total_seq_len=static_cast<cl_int>(total_seq_len);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_seq_number); 
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_total_seq_len); 

    // 提交到kernel执行队列 (2D: seq_len x batch_size)
    size_t global_work_size[2] = {(size_t)seq_number,(size_t)batch_size};
    clerr = clEnqueueNDRangeKernel(cl_queue,kernel,2,nullptr,global_work_size,nullptr,0,nullptr,nullptr);

    // 确保执行完成后再进行可能的数据回传
    clFinish(cl_queue);

    if(y_svm)
    {
        size_t num_elems =
            (batch_size - 1) * y_stride_batch +
            (seq_number - 1) * y_stride_i +
            (total_seq_len - 1) * y_stride_j + 1;
        infinirtMemcpy(y,y_svm,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_D2H);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    // 获取opencl后端设备
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
    CHECK_STATUS(launchKernel(_info,y,x,clcontext,cldevice,clqueue));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::opencl