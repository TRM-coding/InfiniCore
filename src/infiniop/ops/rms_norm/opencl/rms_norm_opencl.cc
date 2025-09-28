#include "rms_norm_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

static const char *RmsNormKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Ta
#define Ta float
#endif

#ifndef Tw
#define Tw float
#endif

#ifndef Tcompute
#define Tcompute float
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 1
#endif

typedef int Tidx;

kernel void rms_norm(
    global Ta *y_,
    int const s_y_batch,
    int const s_y_nhead,
    global Ta const *x_,
    int const s_x_batch,
    int const s_x_nhead,
    global Tw const *w,
    float const epsilon,
    Tidx const nhead,
    Tidx const d) {

    Tidx g_idx = get_group_id(0),
         l_idx = get_local_id(0),
         l_len = get_local_size(0);
    Tidx batch_id = g_idx / nhead,
         nhead_id = g_idx % nhead;
    global Ta
        *y = y_ + batch_id * s_y_batch + nhead_id * s_y_nhead;
    global Ta const
        *x = x_ + batch_id * s_x_batch + nhead_id * s_x_nhead;

    Tcompute val_x[ITEMS_THREAD];
    Tcompute val_w[ITEMS_THREAD];
    Tcompute squared = 0;
    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len) {
        val_x[i] = (Tcompute)x[idx];
        val_w[i] = (Tcompute)w[idx];
        squared += val_x[i] * val_x[i];
    }
    // TODO:测试加载相邻元素处理；
    Tcompute mean_sq = work_group_reduce_add(squared) / (Tcompute)d;
    Tcompute rms = native_rsqrt(mean_sq + (Tcompute)epsilon);

    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len)
        y[idx] = (Ta)(rms * val_x[i] * val_w[i]);
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

namespace op::rms_norm::opencl {

struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// launch kernel
infiniStatus_t launchKernel(
    uint32_t batch_size, size_t nhead, size_t dim,
    void *y, infiniDtype_t atype, ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    const void *x, ptrdiff_t stride_x_batch, ptrdiff_t stride_x_nhead,
    const void *w, infiniDtype_t wtype,
    float epsilon,
    size_t block_size,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {
    std::string dt_a, dt_w, dt_compute;
    dt_compute = "float";
    if (!dtypeToClType(atype, dt_a)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtypeToClType(wtype, dt_w)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    size_t items_perthread = (dim + block_size - 1) / block_size;

    const char *src_ptr = RmsNormKernelSource;
    size_t src_len = std::strlen(src_ptr);

    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);
    if (clerr != CL_SUCCESS || program == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Ta=" + dt_a + " ";
    build_opts += "-D Tw=" + dt_w + " ";
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

    cl_kernel kernel = clCreateKernel(program, "rms_norm", &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *y_svm = NULL;
    void *x_svm = NULL;
    void *w_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) {
        infinirtMalloc(&y_svm, ((batch_size - 1) * stride_y_batch + (nhead - 1) * stride_y_nhead + dim) * dtypeSize(atype));
        infinirtMemcpy(y_svm, y, ((batch_size - 1) * stride_y_batch + (nhead - 1) * stride_y_nhead + dim) * dtypeSize(atype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }
    cl_int s_y_batch = static_cast<cl_int>(stride_y_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_batch);
    cl_int s_y_nhead = static_cast<cl_int>(stride_y_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_nhead);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) {
        infinirtMalloc(&x_svm, ((batch_size - 1) * stride_x_batch + (nhead - 1) * stride_x_nhead + dim) * dtypeSize(atype));
        infinirtMemcpy(x_svm, x, ((batch_size - 1) * stride_x_batch + (nhead - 1) * stride_x_nhead + dim) * dtypeSize(atype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }
    printf("%d , %d , %d, \n", batch_size, static_cast<int>(stride_y_batch), static_cast<int>(stride_x_batch));
    cl_int s_x_batch = static_cast<cl_int>(stride_x_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_x_batch);
    cl_int s_x_nhead = static_cast<cl_int>(stride_x_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_x_nhead);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, w);
    if (clerr != CL_SUCCESS) {
        infinirtMalloc(&w_svm, dim * dtypeSize(wtype));
        infinirtMemcpy(w_svm, w, dim * dtypeSize(wtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, w_svm);
    }
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &epsilon);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &dim);

    size_t global_size = batch_size * nhead * block_size;

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &global_size, &block_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size, block_size);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (y_svm) {
        infinirtMemcpy(y, y_svm, ((batch_size - 1) * stride_y_batch + (nhead - 1) * stride_y_nhead + dim) * dtypeSize(atype), INFINIRT_MEMCPY_D2H);
        infinirtFree(y_svm);
    }
    if (x_svm) {
        infinirtFree(x_svm);
    }
    if (w_svm) {
        infinirtFree(w_svm);
    }

    // cleanup program/kernel
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {
    // std::cout<<"RMS_NORM Running"<<std::endl;

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stride_x_batch = _info.x_strides[0];
    auto stride_x_nhead = _info.x_strides[1];
    auto stride_y_batch = _info.y_strides[0];
    auto stride_y_nhead = _info.y_strides[1];
    auto dim = _info.dim();
    uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);
    size_t nhead = _info.shape.size() > 2 ? _info.shape[1] : 1;
    size_t block_size = _opaque->internal->maxThreadsPerBlock();
    void *device;
    void *context;

    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));

    auto device_cl = reinterpret_cast<cl_device_id>(device);
    cl_context context_cl = reinterpret_cast<cl_context>(context);

    cl_uint num_devices;
    auto err_c = clGetContextInfo(context_cl, CL_CONTEXT_NUM_DEVICES, sizeof(num_devices), &num_devices, nullptr);
    if (err_c != CL_SUCCESS) {
        std::cerr << "Error getting context device count!" << std::endl;
    } 
    // else {
    //     std::cout << "Number of Devices in Context: " << num_devices << std::endl;
    // }

    // 获取上下文中的设备列表
    cl_device_id *devices_in_context = new cl_device_id[num_devices];
    err_c = clGetContextInfo(context_cl, CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), devices_in_context, nullptr);
    if (err_c != CL_SUCCESS) {
        std::cerr << "Error getting devices in context!" << std::endl;
    } 
    // else {
    //     // std::cout << "Devices in Context:" << std::endl;
    //     for (cl_uint i = 0; i < num_devices; ++i) {
    //         char device_name[1024];
    //         err_c = clGetDeviceInfo(devices_in_context[i], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    //         if (err_c != CL_SUCCESS) {
    //             std::cerr << "Error getting device name!" << std::endl;
    //         } 
    //         // else {
    //         //     std::cout << "Device " << i + 1 << ": " << device_name << std::endl;
    //         // }
    //     }
    // }

    char device_name[1024];
    auto err = clGetDeviceInfo(device_cl, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting device name!" << std::endl;
    } 
    // else {
    //     std::cout << "Device Name: " << device_name << std::endl;
    // }

    cl_context clcontext = static_cast<cl_context>(context);
    cl_device_id cldevice = static_cast<cl_device_id>(device);
    if (!stream) {
        CHECK_STATUS(infinirtGetOpenclStream(&stream));
    }
    cl_command_queue clqueue = static_cast<cl_command_queue>(stream);
    CHECK_STATUS(launchKernel(batch_size, nhead, dim, y, _info.atype, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, _info.wtype, _info.epsilon, block_size, clcontext, cldevice, clqueue));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::opencl
