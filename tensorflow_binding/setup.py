import os
import setuptools
import tensorflow as tf
from setuptools.command.build_ext import build_ext


CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

include_dirs = [tf.sysconfig.get_include(), os.path.join(CUDA_HOME, "include")]

core_dir = os.path.realpath("./build")

extra_link_args = ["-L" + tf.sysconfig.get_lib()]

extra_compile_args = ["-std=c++14", "-fPIC", "-Wno-return-type"]

ext = setuptools.Extension("warp_rnnt_tf.kernels",
                           sources=["binding.cpp"],
                           language="c++",
                           include_dirs=include_dirs,
                           library_dirs=[core_dir],
                           runtime_library_dirs=[core_dir],
                           libraries=["warp_rnnt_core", "tensorflow_framework"],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args)

setuptools.setup(
    name="warp_rnnt_tf",
    version="0.2",
    description="TensorFlow wrapper for warp-rnnt",
    license="MIT",
    author = "Lucky Wong",
    packages=["warp_rnnt_tf"],
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
