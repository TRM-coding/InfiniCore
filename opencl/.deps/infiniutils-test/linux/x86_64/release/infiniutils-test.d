{
    files = {
        "pencl/.objs/infiniutils-test/linux/x86_64/release/src/utils-test/main.cc.o",
        "pencl/.objs/infiniutils-test/linux/x86_64/release/src/utils-test/test_rearrange.cc.o",
        "pencl/linux/x86_64/release/libinfini-utils.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-Lpencl/linux/x86_64/release",
            "-s",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}