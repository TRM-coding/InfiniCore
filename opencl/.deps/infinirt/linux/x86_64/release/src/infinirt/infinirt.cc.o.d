{
    files = {
        "src/infinirt/infinirt.cc"
    },
    depfiles = "infinirt.o: src/infinirt/infinirt.cc include/infinirt.h  include/infinicore.h src/infinirt/../utils.h  src/infinirt/../utils/custom_types.h src/infinirt/../utils/rearrange.h  src/infinirt/../utils/result.hpp src/infinirt/../utils/check.h  include/infinicore.h src/infinirt/ascend/infinirt_ascend.h  src/infinirt/ascend/../infinirt_impl.h src/infinirt/bang/infinirt_bang.h  src/infinirt/bang/../infinirt_impl.h src/infinirt/cpu/infinirt_cpu.h  src/infinirt/cpu/../infinirt_impl.h src/infinirt/cuda/infinirt_cuda.cuh  src/infinirt/cuda/../infinirt_impl.h  src/infinirt/kunlun/infinirt_kunlun.h  src/infinirt/kunlun/../infinirt_impl.h  src/infinirt/metax/infinirt_metax.h  src/infinirt/metax/../infinirt_impl.h  src/infinirt/moore/infinirt_moore.h  src/infinirt/moore/../infinirt_impl.h  src/infinirt/opencl/infinirt_opencl.h  src/infinirt/opencl/../infinirt_impl.h\
",
    depfiles_format = "gcc",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    }
}