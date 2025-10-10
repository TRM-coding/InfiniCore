#include "gemm_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "infiniop/handle.h"
#include "infinirt.h"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include <chrono>

static const char *GemmKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifndef T
#define T float
#endif

#ifndef Tcompute
#define Tcompute float
#endif


kernel void gemm_kernel(
    global T *C,
    int const c_row_stride,
    int const c_col_stride,
    global T const *A,
    int const a_row_stride,
    int const a_col_stride,
    global T const *B,
    int const b_row_stride,
    int const b_col_stride,
    float const alpha,
    float const beta,
    int const M,
    int const N,
    int const K,
    int const batch_stride_a,
    int const batch_stride_b,
    int const batch_stride_c) {

    int i = get_group_id(0);
    int j = get_group_id(1);
    int b = get_group_id(2);

    if (i >= M || j >= N) return;

    size_t baseA = (size_t)b * (size_t)batch_stride_a + (size_t)i * (size_t)a_row_stride;
    size_t baseB = (size_t)b * (size_t)batch_stride_b + (size_t)j * (size_t)b_col_stride;
    size_t idxC  = (size_t)b * (size_t)batch_stride_c + (size_t)i * (size_t)c_row_stride + (size_t)j * (size_t)c_col_stride;

    uint lane = get_sub_group_local_id();
    uint sg   = get_sub_group_size();

    Tcompute acc = (Tcompute)0;
    for (int k = (int)lane; k < K; k += (int)sg) {
        T a = A[baseA + (size_t)k * (size_t)a_col_stride];   // A(i, k)
        T bt = B[baseB + (size_t)k * (size_t)b_row_stride];  // B(k, j)
        acc += (Tcompute)a * (Tcompute)bt;
    }

    Tcompute sum = sub_group_reduce_add(acc);

    if (lane == 0) {
        Tcompute out = (Tcompute)alpha * sum;
        if (beta != 0.0f) {
            out += (Tcompute)beta * (Tcompute)C[idxC];
        }
        C[idxC] = (T)out;
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

namespace op::gemm::opencl {

struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
    cl_program program_cache=NULL;
    cl_kernel kernel_cache=NULL;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    
    auto dtype = c_desc->dtype();
    if (a_desc->dtype() != dtype || b_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);
    auto info = result.take();
    
    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        0, 
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        handle->device, 
        handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

// Launch GEMM kernel
infiniStatus_t launchKernel(
    const MatmulInfo &info,
    infiniDtype_t dtype,
    void *c, const void *a, const void *b,
    float alpha, float beta,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue,
    cl_kernel& kernel,
    cl_program& program) {

    //获取算子基本元数据
    auto batch_size=info.batch;
    auto a_row_size=info.a_matrix.rows;
    auto a_col_size=info.a_matrix.cols;
    auto a_row_stride=info.a_matrix.row_stride;
    auto a_col_stride=info.a_matrix.col_stride;
    auto a_batch_stride=info.a_matrix.stride;

    auto b_row_size=info.b_matrix.rows;
    auto b_col_size=info.b_matrix.cols;
    auto b_row_stride=info.b_matrix.row_stride;
    auto b_col_stride=info.b_matrix.col_stride;
    auto b_batch_stride=info.b_matrix.stride;

    auto c_row_size=info.c_matrix.rows;
    auto c_col_size=info.c_matrix.cols;
    auto c_row_stride=info.c_matrix.row_stride;
    auto c_col_stride=info.c_matrix.col_stride;
    auto c_batch_stride=info.c_matrix.stride;

    auto M=info.m;//M 行
    auto N=info.n;//N 列
    auto K=info.k;//中间维度

    

    //数值类型转换
    std::string dt,dt_compute;
    dt_compute="float";
    dtypeToClType(dtype,dt);

    //创建程序对象
    const char * src_ptr = GemmKernelSource;
    size_t src_len = std::strlen(src_ptr);
    cl_int clerr;
    if(program==NULL){
        program = clCreateProgramWithSource(context,1,&src_ptr,&src_len,&clerr);
        // std::cout<<std::endl<<"create gemm cache"<<std::endl;
        //构造编译命令并完成编译
        std::string build_opts;
        build_opts += "-D T=" + dt + " ";
        build_opts += "-D Tcompute=" + dt_compute + " ";
        build_opts += "-cl-std=CL2.0 ";
        clerr=clBuildProgram(program,1,&device,build_opts.c_str(),nullptr,nullptr);
    }
    //获取内核代码
    if(kernel==NULL){
        kernel = clCreateKernel(program,"gemm_kernel",&clerr); 
    }
    int arg_idx=0;
    //C矩阵参数传入/////////////////////////////////////////////////////////////////

    //分配参数*C共享内存
    void *c_svm=NULL;
    clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,c);
    if(clerr != CL_SUCCESS)
    {
        // std::cout<<"error:"<<clerr<<std::endl;
        size_t num_elems =
            (batch_size - 1) * c_batch_stride +
            (c_row_size - 1) * c_row_stride +
            (c_col_size - 1) * c_col_stride + 1;
        infinirtMalloc(&c_svm,num_elems*dtypeSize(dtype));
        infinirtMemcpy(c_svm,c,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,c_svm);
    }
    //传入参数C_row_stride,C_col_stride
    cl_int cl_c_row_stride=static_cast<cl_int>(c_row_stride);
    cl_int cl_c_col_stride=static_cast<cl_int>(c_col_stride);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_c_row_stride);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_c_col_stride);
    
    
    //A矩阵参数传入//////////////////////////////////////////////////////////////////////////

    //分配参数*A共享内存
    void *a_svm=NULL;
    clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,a);
    if(clerr != CL_SUCCESS)
    {
        // std::cout<<clerr<<std::endl;
        size_t num_elems =
            (batch_size - 1) * a_batch_stride +
            (a_row_size - 1) * a_row_stride +
            (a_col_size - 1) * a_col_stride + 1;
        // std::cout<<std::endl<<"SVM_failed"<<std::endl;
        infinirtMalloc(&a_svm,num_elems*dtypeSize(dtype));
        infinirtMemcpy(a_svm,a,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,a_svm);
    }
    //传入参数A_row_stride,A_col_stride
    cl_int cl_a_row_stride=static_cast<cl_int>(a_row_stride);
    cl_int cl_a_col_stride=static_cast<cl_int>(a_col_stride);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_a_row_stride);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_a_col_stride);

    //B矩阵参数传入//////////////////////////////////////////////////////////////////////////

    //分配参数*B共享内存
    void *b_svm=NULL;
    clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,b);
    if(clerr != CL_SUCCESS)
    {
        // std::cout<<clerr<<std::endl;
        size_t num_elems =
            (batch_size - 1) * b_batch_stride +
            (b_row_size - 1) * b_row_stride +
            (b_col_size - 1) * b_col_stride + 1;
        infinirtMalloc(&b_svm,num_elems*dtypeSize(dtype));
        infinirtMemcpy(b_svm,b,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel,arg_idx++,b_svm);
    }
    //传入参数B_row_stride,B_col_stride
    cl_int cl_b_row_stride=static_cast<cl_int>(b_row_stride);
    cl_int cl_b_col_stride=static_cast<cl_int>(b_col_stride);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_b_row_stride);
    clerr |= clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_b_col_stride);

    //alpha,beta,M,N,K,batch_stride_a,batch_stride_b,batch_stride_c传入////////////
    cl_float cl_alpha=static_cast<cl_float>(alpha);
    cl_float cl_beta =static_cast<cl_float>(beta);
    cl_int cl_M = static_cast<cl_int>(M);
    cl_int cl_N = static_cast<cl_int>(N);
    cl_int cl_K = static_cast<cl_int>(K);
    cl_int cl_batch_stride_a = static_cast<cl_int>(a_batch_stride);
    cl_int cl_batch_stride_b = static_cast<cl_int>(b_batch_stride);
    cl_int cl_batch_stride_c = static_cast<cl_int>(c_batch_stride);

    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_float),&cl_alpha);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_float),&cl_beta);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_M);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_N);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_K);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_batch_stride_a);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_batch_stride_b);
    clerr |=clSetKernelArg(kernel,arg_idx++,sizeof(cl_int),&cl_batch_stride_c);
    
    // 选择本地/全局工作尺寸以匹配子组计算
    // 使用首选工作组倍数作为子组大小的近似
    size_t preferred_multiple = 0;
    clerr = clGetKernelWorkGroupInfo(kernel, device,
                                     CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                     sizeof(preferred_multiple), &preferred_multiple, nullptr);
    if (clerr != CL_SUCCESS || preferred_multiple == 0) {
        preferred_multiple = 1; // fallback
    }

    // std::cout<<"work_gourp:"<<preferred_multiple<<std::endl;

    size_t local_work_size[3]  = { preferred_multiple, 1, 1 };
    size_t global_work_size[3] = { (size_t)M * local_work_size[0],
                                   (size_t)N,
                                   (size_t)batch_size };

    //提交到kernel执行队列
    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 3, nullptr,
                                   global_work_size, local_work_size,
                                   0, nullptr, nullptr);

    if(c_svm)
    {
        size_t num_elems =
            (batch_size - 1) * c_batch_stride +
            (c_row_size - 1) * c_row_stride +
            (c_col_size - 1) * c_col_stride + 1;
        infinirtMemcpy(c,c_svm,num_elems*dtypeSize(dtype),INFINIRT_MEMCPY_D2H);
        infinirtFree(c_svm);
    }

    // clReleaseKernel(kernel);
    // clReleaseProgram(program);
    if (a_svm) {
        infinirtFree(a_svm);
    }
    if (b_svm) {
        infinirtFree(b_svm);
    }
    // std::cout<<"GEMM Runing Finished"<<std::endl;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {
    if (_info.is_transed) {
        std::swap(a, b);
    }
    using clock = std::chrono::steady_clock;  
    auto t0 = clock::now();
    // std::cout<<"GEMM Running"<<std::endl;
    void *device;
    void *context;

    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));

    auto device_cl = reinterpret_cast<cl_device_id>(device);
    auto context_cl = reinterpret_cast<cl_context>(context);

    //获取context中的设备数量
    cl_uint num_devices;
    auto err_c = clGetContextInfo(context_cl,CL_CONTEXT_NUM_DEVICES,sizeof(num_devices),&num_devices,nullptr);

    //获取context中的设别列表
    cl_device_id *devices_in_context = new cl_device_id[num_devices];
    err_c = clGetContextInfo(context_cl,CL_CONTEXT_DEVICES,num_devices*sizeof(cl_device_id),devices_in_context,nullptr);


    auto clcontext = static_cast<cl_context>(context);
    auto cldevice = static_cast <cl_device_id>(device);

    if(!stream)
    {
        CHECK_STATUS(infinirtGetOpenclStream(&stream));
    }
    auto clqueue = static_cast<cl_command_queue>(stream);
    auto& kernel_cache=this->_opaque->kernel_cache;
    auto& program_cache=this->_opaque->program_cache;
    CHECK_STATUS(launchKernel(_info,_dtype,c,a,b,alpha,beta,clcontext,cldevice,clqueue,kernel_cache,program_cache));
    auto t1 = clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "GEMM_time: " << ms/1000.0 << " ms\n";
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::opencl
