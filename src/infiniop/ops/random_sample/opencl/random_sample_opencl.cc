#include "random_sample_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "infiniop/handle.h"
#include "infinirt.h"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <iostream>


static const char *RandomSampleKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef SCALAR_T
#define SCALAR_T float
#endif

#ifndef COMPUTE_T
#define COMPUTE_T float
#endif

kernel void random_sample_kernel(
    global int* result,
    global const SCALAR_T* probs,  
    float random_val,
    float topp,
    int topk,
    float temperature,
    int n
) {
    int N = n;
    if (N <= 0) {
        if (result) result[0] = 0;
        return;
    }

    // 1) Find max(probs)
    COMPUTE_T max_val = (COMPUTE_T)(-INFINITY);
    for (int i = 0; i < N; ++i) {
        COMPUTE_T v = (COMPUTE_T)probs[i];
        if (v > max_val) max_val = v;
    }

    // 2) Total softmax sum with temperature
    COMPUTE_T inv_temp = (COMPUTE_T)1.0f / (COMPUTE_T)temperature; // follows CPU semantics (division as-is)
    COMPUTE_T total_sum = (COMPUTE_T)0;
    for (int i = 0; i < N; ++i) {
        COMPUTE_T v = (COMPUTE_T)probs[i];
        total_sum += exp((v - max_val) * inv_temp);
    }

    // 3) Determine k consistent with CPU semantics
    int k = topk;
    if (k <= 0 || k > N) k = N;

    // Helper state for stable descending selection by value, tie by index
    COMPUTE_T prev_val = (COMPUTE_T)(INFINITY);
    int last_idx = -1;

    // 4) Compute pk = cumulative sum up to k-th largest
    COMPUTE_T pk = (COMPUTE_T)0;
    for (int t = 0; t < k; ++t) {
        COMPUTE_T best_val = (COMPUTE_T)(-INFINITY);
        int best_idx = -1;

        for (int i = 0; i < N; ++i) {
            COMPUTE_T vi = (COMPUTE_T)probs[i];
            int eligible = (vi < prev_val) || ((vi == prev_val) && (i > last_idx));
            if (!eligible) continue;

            if (best_idx < 0 || vi > best_val || (vi == best_val && i < best_idx)) {
                best_val = vi;
                best_idx = i;
            }
        }

        if (best_idx < 0) break; // safety
        pk += exp((best_val - max_val) * inv_temp);
        prev_val = best_val;
        last_idx = best_idx;
    }

    // 5) Compute plimit
    COMPUTE_T pp = total_sum * (COMPUTE_T)topp;
    COMPUTE_T min_pk_pp = (pk < pp) ? pk : pp;
    COMPUTE_T plimit = (COMPUTE_T)random_val * min_pk_pp;

    // 6) Second pass: sample first index where cumulative >= plimit
    prev_val = (COMPUTE_T)(INFINITY);
    last_idx = -1;
    COMPUTE_T cumsum = (COMPUTE_T)0;
    int out_idx = 0; // default

    for (int t = 0; t < k; ++t) {
        COMPUTE_T best_val = (COMPUTE_T)(-INFINITY);
        int best_idx = -1;

        for (int i = 0; i < N; ++i) {
            COMPUTE_T vi = (COMPUTE_T)probs[i];
            int eligible = (vi < prev_val) || ((vi == prev_val) && (i > last_idx));
            if (!eligible) continue;

            if (best_idx < 0 || vi > best_val || (vi == best_val && i < best_idx)) {
                best_val = vi;
                best_idx = i;
            }
        }

        if (best_idx < 0) break; // safety
        cumsum += exp((best_val - max_val) * inv_temp);
        if (plimit <= cumsum) {
            out_idx = best_idx;
            break;
        }
        prev_val = best_val;
        last_idx = best_idx;
        out_idx = best_idx; // fallback if plimit hits after last element due to FP rounding
    }

    result[0] = out_idx;
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

namespace op::random_sample::opencl {
struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
};
Descriptor::~Descriptor() {}
size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);
    // std::cout<<"start create"<<std::endl;
    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle_)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    const RandomSampleInfo &info,
    void * result,
    void const* probs,
    float random_val,
    float topp,
    int topk,
    float temperature,    
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue) {

    //获取算子基本元数据
    auto dtype_in = info.dt_p;
    auto dtype_out = info.dt_i;
    int sample_len = info.n;

    //数值类型转换
    std::string dt, dt_compute;
    dt_compute = "float";
    if (!dtypeToClType(dtype_in, dt)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    //创建程序对象
    const char * src_ptr = RandomSampleKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr;
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);
    if (clerr != CL_SUCCESS || program == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    //构造编译命令并完成编译
    std::string build_opts;
    build_opts += "-D SCALAR_T=" + dt + " ";
    build_opts += "-D COMPUTE_T=" + dt_compute + " ";
    build_opts += "-cl-std=CL2.0 ";
    clerr = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        // 打印构建日志，便于定位问题
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            log[log_size] = '\0';
            fprintf(stderr, "[OpenCL] random_sample build log:\n%s\n", log.data());
        }
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    //获取内核代码
    cl_kernel kernel = clCreateKernel(program, "random_sample_kernel", &clerr); 
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;

    // result 传入 - 优先尝试直接指针，失败则分配SVM并拷贝
    void *result_svm = nullptr;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, result);
    if (clerr != CL_SUCCESS) {
        size_t num_elems = 1;  // result只有一个元素
        infinirtMalloc(&result_svm, num_elems * sizeof(int));
        infinirtMemcpy(result_svm, result, num_elems * sizeof(int), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, result_svm);
        if (clerr != CL_SUCCESS) {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    // probs 传入 - 修正为先传原始指针
    void *probs_svm = nullptr;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, const_cast<void*>(probs));
    if (clerr != CL_SUCCESS) {
        size_t num_elems = (size_t)sample_len;
        infinirtMalloc(&probs_svm, num_elems * dtypeSize(dtype_in));
        infinirtMemcpy(probs_svm, probs, num_elems * dtypeSize(dtype_in), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, probs_svm);
        if (clerr != CL_SUCCESS) {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    // random_val, topp, topk, temperature, n 传入
    cl_float cl_random_val = static_cast<cl_float>(random_val);
    cl_float cl_topp       = static_cast<cl_float>(topp);
    cl_int   cl_topk       = static_cast<cl_int>(topk);
    cl_float cl_temperature= static_cast<cl_float>(temperature);
    cl_int   cl_n          = static_cast<cl_int>(sample_len);

    clerr  = clSetKernelArg(kernel, arg_idx++, sizeof(cl_float), &cl_random_val);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_float), &cl_topp);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int),   &cl_topk);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_float), &cl_temperature);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int),   &cl_n);
    if (clerr != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 提交到kernel执行队列
    size_t global_work_size[1] = {1};
    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 确保kernel完成
    clFinish(cl_queue);

    // 拷回结果（当使用了临时SVM时）
    if (result_svm) {
        size_t num_elems = 1;
        infinirtMemcpy(result, result_svm, num_elems * dtypeSize(dtype_out), INFINIRT_MEMCPY_D2H);
        infinirtFree(result_svm);
    }
    if (probs_svm) {
        infinirtFree(probs_svm);
    }

    std::cout << "excute finished" << std::endl;

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
    
    // std::cout<<"RANDOM_SAMPLE Running"<<std::endl;
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
    CHECK_STATUS(launchKernel(_info,result,probs,random_val,topp,topk,temperature,clcontext,cldevice,clqueue));
    return INFINI_STATUS_SUCCESS;
}


} // namespace op::random_sample::opencl