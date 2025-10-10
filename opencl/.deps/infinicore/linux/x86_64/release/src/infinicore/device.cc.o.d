{
    files = {
        "src/infinicore/device.cc"
    },
    depfiles = "device.o: src/infinicore/device.cc include/infinicore.hpp  include/infinicore/tensor.hpp include/infinicore/device.hpp  include/infinicore/dtype.hpp include/infinicore.h\
",
    depfiles_format = "gcc",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fPIC",
            "-O3",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-isystem",
            "/home/tianruiming/.xmake/packages/p/pybind11/v3.0.1/8f5d512d4fdb4713bf705395b25be885/include",
            "-isystem",
            "/home/tianruiming/miniconda3/envs/infini/include/python3.10",
            "-DNDEBUG"
        }
    }
}