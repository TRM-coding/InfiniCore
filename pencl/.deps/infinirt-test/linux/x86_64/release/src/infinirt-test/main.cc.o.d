{
    files = {
        "src/infinirt-test/main.cc"
    },
    depfiles = "main.o: src/infinirt-test/main.cc src/infinirt-test/test.h  src/infinirt-test/../utils.h src/infinirt-test/../utils/custom_types.h  src/infinirt-test/../utils/rearrange.h  src/infinirt-test/../utils/result.hpp src/infinirt-test/../utils/check.h  include/infinicore.h include/infinirt.h include/infinicore.h\
",
    depfiles_format = "gcc",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-Werror",
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