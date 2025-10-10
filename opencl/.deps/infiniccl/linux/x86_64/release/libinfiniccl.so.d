{
    files = {
        "pencl/.objs/infiniccl/linux/x86_64/release/src/infiniccl/infiniccl.cc.o",
        "pencl/linux/x86_64/release/libinfini-utils.a",
        "pencl/linux/x86_64/release/libinfinirt-cpu.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-shared",
            "-m64",
            "-fPIC",
            "-Lpencl/linux/x86_64/release",
            "-s",
            "-linfinirt",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}