{
    files = {
        "pencl/.objs/infinirt-test/linux/x86_64/release/src/infinirt-test/main.cc.o",
        "pencl/.objs/infinirt-test/linux/x86_64/release/src/infinirt-test/test.cc.o",
        "pencl/linux/x86_64/release/libinfini-utils.a",
        "pencl/linux/x86_64/release/libinfinirt-cpu.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-Lpencl/linux/x86_64/release",
            "-Wl,-rpath=$ORIGIN",
            "-s",
            "-linfinirt",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}