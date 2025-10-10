{
    files = {
        "pencl/.objs/infinirt/linux/x86_64/release/src/infinirt/infinirt.cc.o",
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
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}