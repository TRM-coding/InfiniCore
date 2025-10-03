#include "rope_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "infiniop/handle.h"
#include "infinirt.h"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

static const char *RopeKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef T
#define T float
#endif

#ifndef Tcompute
#define Tcompute float
#endif

#ifndef Tpos
#define Tpos int
#endif

kernel void rope_kernel(
    global T *y,
    global const T *x,
    global const Tpos *pos_ids,
    global const T *sin_table,
    global const T *cos_table,
    int const y_stride_seqlen,
    int const x_stride_seqlen,
    int const y_stride_nhead,
    int const x_stride_nhead,
    int const table_dim,
    int const nhead,
    int const seqlen,
    int const is_gpt_j
)
{
    int tok = get_global_id(0);
    int h   = get_global_id(1);
    int i   = get_global_id(2);

    if (tok >= seqlen || h >= nhead || i >= table_dim)
        return;

    size_t x_offset = (size_t)tok * (size_t)x_stride_seqlen + (size_t)h * (size_t)x_stride_nhead;
    size_t y_offset = (size_t)tok * (size_t)y_stride_seqlen + (size_t)h * (size_t)y_stride_nhead;

    size_t pos0 = is_gpt_j ? (size_t)(2 * i) : (size_t)i;
    size_t pos1 = is_gpt_j ? pos0 + 1 : pos0 + (size_t)table_dim;

    T x0T = x[x_offset + pos0];
    T x1T = x[x_offset + pos1];

    size_t pos_id = (size_t)pos_ids[tok];
    size_t table_offset = pos_id * (size_t)table_dim;

    T sinT = sin_table[table_offset + (size_t)i];
    T cosT = cos_table[table_offset + (size_t)i];

    Tcompute x0 = (Tcompute)x0T;
    Tcompute x1 = (Tcompute)x1T;
    Tcompute s  = (Tcompute)sinT;
    Tcompute c  = (Tcompute)cosT;

    Tcompute y0 = x0 * c - x1 * s;
    Tcompute y1 = x0 * s + x1 * c;

    y[y_offset + pos0] = (T)y0;
    y[y_offset + pos1] = (T)y1;
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

// 支持 pos_ids 的整型到 OpenCL 标量类型映射
static bool dtypeToClIndex(infiniDtype_t dt, std::string &out) {
    switch (dt) {
    case INFINI_DTYPE_U8:  out = "uchar";  return true;
    case INFINI_DTYPE_I8:  out = "char";   return true;
    case INFINI_DTYPE_U16: out = "ushort"; return true;
    case INFINI_DTYPE_I16: out = "short";  return true;
    case INFINI_DTYPE_U32: out = "uint";   return true;
    case INFINI_DTYPE_I32: out = "int";    return true;
    case INFINI_DTYPE_U64: out = "ulong";  return true;
    case INFINI_DTYPE_I64: out = "long";   return true;
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

namespace op::rope::opencl {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopRoPEAlgo_t algo
) {

    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);

    auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc,algo);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    const RoPEInfo &info,
    infiniDtype_t dtype,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {
    auto y_stride_seqlen = info.y_stride_seqlen;
    auto x_stride_seqlen = info.x_stride_seqlen;
    auto y_stride_nhead = info.y_stride_nhead;
    auto x_stride_nhead = info.x_stride_nhead;
    auto table_dim = info.table_dim;
    auto nhead = info.nhead;
    auto seqlen = info.seqlen;

    std::string dt, dt_compute;
    dt_compute = "float";
    dtypeToClType(dtype, dt);
    // 新增：pos_ids 对应 OpenCL 类型
    std::string dt_pos = "int";
    dtypeToClIndex(info.pos_type, dt_pos);

    // 创建程序对象
    const char *src_ptr = RopeKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);
    if (clerr != CL_SUCCESS || program == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 构造编译命令并完成编译
    std::string build_opts;
    build_opts += "-D T=" + dt + " ";
    build_opts += "-D Tcompute=" + dt_compute + " ";
    build_opts += "-D Tpos=" + dt_pos + " ";
    build_opts += "-cl-std=CL2.0 ";
    clerr = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            log[log_size] = '\0';
            printf("OpenCL build log (rope): %s\n", log.data());
        }
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 获取内核代码
    cl_kernel kernel = clCreateKernel(program, "rope_kernel", &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    int arg_idx = 0;

    // Y 参数传入
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) {
        size_t y_num_elems = (seqlen - 1) * y_stride_seqlen + (nhead - 1) * y_stride_nhead + (2 * table_dim - 1) + 1;
        infinirtMalloc(&y_svm, y_num_elems * dtypeSize(dtype));
        infinirtMemcpy(y_svm, y, y_num_elems * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }

    // X 参数传入
    void *x_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) {
        size_t x_num_elems = (seqlen - 1) * x_stride_seqlen + (nhead - 1) * x_stride_nhead + (2 * table_dim - 1) + 1;
        infinirtMalloc(&x_svm, x_num_elems * dtypeSize(dtype));
        infinirtMemcpy(x_svm, x, x_num_elems * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }

    // pos_ids 传入
    void *pos_ids_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, pos_ids);
    if (clerr != CL_SUCCESS) {
        size_t pos_ids_num_elems = seqlen;
        infinirtMalloc(&pos_ids_svm, pos_ids_num_elems * dtypeSize(info.pos_type));
        infinirtMemcpy(pos_ids_svm, pos_ids, pos_ids_num_elems * dtypeSize(info.pos_type), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, pos_ids_svm);
    }

    // sin_table 传入
    void *sin_table_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, sin_table);
    if (clerr != CL_SUCCESS) {
        size_t sin_table_num_elems = seqlen * table_dim;
        infinirtMalloc(&sin_table_svm, sin_table_num_elems * dtypeSize(dtype));
        infinirtMemcpy(sin_table_svm, sin_table, sin_table_num_elems * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, sin_table_svm);
    }

    // cos_table 传入
    void *cos_table_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, cos_table);
    if (clerr != CL_SUCCESS) {
        size_t cos_table_num_elems = seqlen * table_dim;
        infinirtMalloc(&cos_table_svm, cos_table_num_elems * dtypeSize(dtype));
        infinirtMemcpy(cos_table_svm, cos_table, cos_table_num_elems * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, cos_table_svm);
    }

    // 其他参数传入
    cl_int cl_y_stride_seqlen = static_cast<cl_int>(y_stride_seqlen);
    cl_int cl_x_stride_seqlen = static_cast<cl_int>(x_stride_seqlen);
    cl_int cl_y_stride_nhead = static_cast<cl_int>(y_stride_nhead);
    cl_int cl_x_stride_nhead = static_cast<cl_int>(x_stride_nhead);
    cl_int cl_table_dim = static_cast<cl_int>(table_dim);
    cl_int cl_nhead = static_cast<cl_int>(nhead);
    cl_int cl_seqlen = static_cast<cl_int>(seqlen);
    cl_int cl_is_gpt_j = info.algo == infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_GPT_J ? 1 : 0;

    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_y_stride_seqlen);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_x_stride_seqlen);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_y_stride_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_x_stride_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_table_dim);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_seqlen);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_is_gpt_j);
    // 设置全局工作尺寸: (seqlen, nhead, table_dim)
    size_t global_work_size[3] = {(size_t)seqlen, (size_t)nhead, (size_t)table_dim};

    // 启动kernel
    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 3, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL][rope] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 如果使用了 SVM 内存进行数据传输，执行数据传输
    if (y_svm) {
        size_t num_elems =
            (seqlen - 1) * y_stride_seqlen +
            (nhead - 1) * y_stride_nhead +
            (2 * table_dim - 1) + 1;
        infinirtMemcpy(y, y_svm, num_elems * dtypeSize(dtype), INFINIRT_MEMCPY_D2H);
        infinirtFree(y_svm);
    }
    if (x_svm) {
        infinirtFree(x_svm);
    }
    if (pos_ids_svm) {
        infinirtFree(pos_ids_svm);
    }
    if (sin_table_svm) {
        infinirtFree(sin_table_svm);
    }
    if (cos_table_svm) {
        infinirtFree(cos_table_svm);
    }

    // 释放资源
    clReleaseKernel(kernel);
    clReleaseProgram(program);
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
    // std::cout<<"ROPE Running"<<std::endl;
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
    CHECK_STATUS(launchKernel(_info, _info.data_type, y, x, pos_ids, sin_table, cos_table, clcontext, cldevice, clqueue));
    return INFINI_STATUS_SUCCESS;
}

#undef ROPE_TYPE
#undef CALCULATE_ROPE

} // namespace op::rope::opencl
