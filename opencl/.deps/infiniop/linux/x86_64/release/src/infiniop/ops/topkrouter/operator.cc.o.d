{
    files = {
        "src/infiniop/ops/topkrouter/operator.cc"
    },
    depfiles = "operator.o: src/infiniop/ops/topkrouter/operator.cc  src/infiniop/ops/topkrouter/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/topkrouter/../../handle.h include/infiniop/handle.h  include/infiniop/ops/topkrouter.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/topkrouter/cpu/topkrouter_cpu.h  src/infiniop/ops/topkrouter/cpu/../topkrouter.h  src/infiniop/ops/topkrouter/cpu/../../../operator.h  src/infiniop/ops/topkrouter/cpu/../info.h  src/infiniop/ops/topkrouter/cpu/../../../../utils.h  src/infiniop/ops/topkrouter/cpu/../../../../utils/custom_types.h  src/infiniop/ops/topkrouter/cpu/../../../../utils/rearrange.h  src/infiniop/ops/topkrouter/cpu/../../../../utils/result.hpp  src/infiniop/ops/topkrouter/cpu/../../../../utils/check.h  include/infinicore.h src/infiniop/ops/topkrouter/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/topkrouter/cpu/../../../../utils.h\
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