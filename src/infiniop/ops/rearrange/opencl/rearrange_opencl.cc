#include "rearrange_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../../../tensor.h"
#include "infiniop/handle.h"
#include "infinirt.h"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

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

static const char *RearrangeKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void rearrange_kernel(
    global char* dst,         // 输出数据指针
    global const char* src,   // 输入数据指针
    int ndim,                 // 维度数
    long count,               // 元素块数量
    int unit,                 // 每个块大小（字节数）
    global const long* idx_strides,  // idx_strides 数组
    global const long* dst_strides,  // dst_strides 数组
    global const long* src_strides   // src_strides 数组
)
{
    size_t gid = get_global_id(0);
    if ((long)gid >= count) return;

    global char* dptr = dst;
    global const char* sptr = src;

    long rem = (long)gid;
    for (int j = 0; j < ndim; ++j) {
        long k = rem / idx_strides[j];
        dptr += k * dst_strides[j];
        sptr += k * src_strides[j];
        rem  = rem % idx_strides[j];
    }

    // 按字节拷贝 unit 大小
    for (int b = 0; b < unit; ++b) {
        dptr[b] = sptr[b];
    }
}
)CLC";

namespace op::rearrange::opencl {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    auto ndim = y_desc->ndim();

    auto y_shape = y_desc->shape();
    auto x_shape = x_desc->shape();
    CHECK_OR_RETURN(x_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() == ndim, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_SAME_SHAPE(x_shape, y_shape);

    auto dst_strides = y_desc->strides();
    auto src_strides = x_desc->strides();
    auto element_size = infiniSizeOf(dtype);

    auto result = utils::RearrangeMeta::create(y_shape.data(), dst_strides.data(), src_strides.data(), ndim, element_size);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        dtype,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    const utils::RearrangeMeta &info,
    infiniDtype_t dtype,
    void *y,
    const void *x,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {

    auto ndim_ = info.ndim();
    auto count_ = info.count();
    auto unit_ = info.unit();
    auto idx_strides_ = info.idx_strides();
    auto dst_strides_ = info.dst_strides();
    auto src_strides_ = info.src_strides();

    // 创建程序对象
    const char *src_ptr = RearrangeKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);

    // 构造编译命令并完成编译
    std::string build_opts;
    build_opts += "-cl-std=CL2.0 ";
    clerr = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);

    // 获取内核代码
    cl_kernel kernel = clCreateKernel(program, "rearrange_kernel", &clerr);
    int arg_idx = 0;


    auto copyHostToSvm = [&](void *svm_ptr, const void *host_ptr, size_t bytes) -> infiniStatus_t {
        if (bytes == 0) {
            return INFINI_STATUS_SUCCESS;
        }
        cl_int err = clEnqueueSVMMap(cl_queue, CL_TRUE, CL_MAP_WRITE, svm_ptr, bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(svm_ptr, host_ptr, bytes);
        err = clEnqueueSVMUnmap(cl_queue, svm_ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        err = clFinish(cl_queue);
        if (err != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    };
    auto copySvmToHost = [&](void *host_ptr, void *svm_ptr, size_t bytes) -> infiniStatus_t {
        if (bytes == 0) {
            return INFINI_STATUS_SUCCESS;
        }
        cl_int err = clEnqueueSVMMap(cl_queue, CL_TRUE, CL_MAP_READ, svm_ptr, bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(host_ptr, svm_ptr, bytes);
        err = clEnqueueSVMUnmap(cl_queue, svm_ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        err = clFinish(cl_queue);
        if (err != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    };

    // y 参数
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = count_ * unit_;
        infinirtMalloc(&y_svm, num_bytes);
        if (copyHostToSvm(y_svm, y, num_bytes) != INFINI_STATUS_SUCCESS) {
            if (y_svm) infinirtFree(y_svm);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }

    // x 参数
    void *x_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = count_ * unit_;
        infinirtMalloc(&x_svm, num_bytes);
        if (copyHostToSvm(x_svm, x, num_bytes) != INFINI_STATUS_SUCCESS) {
            if (y_svm) infinirtFree(y_svm);
            if (x_svm) infinirtFree(x_svm);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }

    cl_int cl_ndim = static_cast<cl_int>(ndim_);
    cl_long cl_count = static_cast<cl_long>(count_);
    cl_int cl_unit = static_cast<cl_int>(unit_);

    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_ndim);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_long), &cl_count);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &cl_unit);

    // idx_strides 参数
    void *idx_strides_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, idx_strides_);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim_ * sizeof(cl_long);
        infinirtMalloc(&idx_strides_svm, num_bytes);
        if (copyHostToSvm(idx_strides_svm, idx_strides_, num_bytes) != INFINI_STATUS_SUCCESS) {
            if (y_svm) infinirtFree(y_svm);
            if (x_svm) infinirtFree(x_svm);
            if (idx_strides_svm) infinirtFree(idx_strides_svm);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, idx_strides_svm);
    }

    // dst_strides 参数
    void *dst_strides_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, dst_strides_);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim_ * sizeof(cl_long);
        infinirtMalloc(&dst_strides_svm, num_bytes);
        if (copyHostToSvm(dst_strides_svm, dst_strides_, num_bytes) != INFINI_STATUS_SUCCESS) {
            if (y_svm) infinirtFree(y_svm);
            if (x_svm) infinirtFree(x_svm);
            if (idx_strides_svm) infinirtFree(idx_strides_svm);
            if (dst_strides_svm) infinirtFree(dst_strides_svm);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, dst_strides_svm);
    }

    // src_strides 参数
    void *src_strides_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, src_strides_);
    if (clerr != CL_SUCCESS) {
        size_t num_bytes = ndim_ * sizeof(cl_long);
        infinirtMalloc(&src_strides_svm, num_bytes);
        if (copyHostToSvm(src_strides_svm, src_strides_, num_bytes) != INFINI_STATUS_SUCCESS) {
            if (y_svm) infinirtFree(y_svm);
            if (x_svm) infinirtFree(x_svm);
            if (idx_strides_svm) infinirtFree(idx_strides_svm);
            if (dst_strides_svm) infinirtFree(dst_strides_svm);
            if (src_strides_svm) infinirtFree(src_strides_svm);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, src_strides_svm);
    }

    // 设置全局工作尺寸：使用 count_ 来决定工作项的数量
    size_t global_work_size[1] = {(size_t)count_};

    // 启动 OpenCL kernel
    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (y_svm) {
        size_t num_bytes = count_ * unit_;
        if (copySvmToHost(y, y_svm, num_bytes) != INFINI_STATUS_SUCCESS) {
            if (y_svm) infinirtFree(y_svm);
            if (x_svm) infinirtFree(x_svm);
            if (idx_strides_svm) infinirtFree(idx_strides_svm);
            if (dst_strides_svm) infinirtFree(dst_strides_svm);
            if (src_strides_svm) infinirtFree(src_strides_svm);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    // 等待执行完成
    

    // 如果使用了 SVM 内存进行数据传输，执行数据传输
    // if (y_svm) {
    //     size_t num_bytes = count_ * unit_;
    //     infinirtMemcpy(y, y_svm, num_bytes, INFINIRT_MEMCPY_D2H);
    // }

    // 释放临时资源
    if (y_svm) infinirtFree(y_svm);
    if (x_svm) infinirtFree(x_svm);
    if (idx_strides_svm) infinirtFree(idx_strides_svm);
    if (dst_strides_svm) infinirtFree(dst_strides_svm);
    if (src_strides_svm) infinirtFree(src_strides_svm);

    // 释放OpenCL对象
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {
    // std::cout<<"REARRANGE Running"<<std::endl;
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
    CHECK_STATUS(launchKernel(_meta, dtype, y, x, clcontext, cldevice, clqueue));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::opencl
