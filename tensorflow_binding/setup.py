"""setup.py script for transducer TensorFlow wrapper"""

from __future__ import print_function

import os
import platform
import re
import setuptools
import sys
import unittest
import warnings
from setuptools.command.build_ext import build_ext as orig_build_ext

if "CUDA_HOME" not in os.environ:
    print("CUDA_HOME not found in the environment so building "
          "without GPU support. To build with GPU support "
          "please define the CUDA_HOME environment variable. "
          "This should be a path which contains include/cuda.h",
          file=sys.stderr)
    CUDA_HOME = '/usr/local/cuda'
else:
    CUDA_HOME = os.environ["CUDA_HOME"]

core_dir = os.path.realpath("./build")

# We need to import tensorflow to find where its include directory is.
try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError(
        "Tensorflow must be installed to build the tensorflow wrapper.")

# /home/huanglk/anaconda3/envs/tensorflow_env/lib/python3.6/site-packages/tensorflow
if "TENSORFLOW_SRC_PATH" not in os.environ:
    print("Please define the TENSORFLOW_SRC_PATH environment variable.\n"
          "This should be a path to the Tensorflow source directory.",
          file=sys.stderr)
    sys.exit(1)

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]

include_dirs = tf_includes + ['./']

if os.getenv("TF_CXX11_ABI") is not None:
    TF_CXX11_ABI = os.getenv("TF_CXX11_ABI")
else:
    warnings.warn("Assuming tensorflow was compiled without C++11 ABI. "
                  "It is generally true if you are using binary pip package. "
                  "If you compiled tensorflow from source with gcc >= 5 and didn't set "
                  "-D_GLIBCXX_USE_CXX11_ABI=0 during compilation, you need to set "
                  "environment variable TF_CXX11_ABI=1 when compiling this bindings. "
                  "Also be sure to touch some files in src to trigger recompilation. "
                  "Also, you need to set (or unsed) this environment variable if getting "
                  "undefined symbol: _ZN10tensorflow... errors")
    TF_CXX11_ABI = "0"

# , '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI
extra_compile_args = ['-std=c++11', '-fPIC']
# current tensorflow code triggers return type errors, silence those for now
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
    extra_link_args = ['-L' + tf.sysconfig.get_lib(), '-ltensorflow_framework']

include_dirs += [os.path.join(CUDA_HOME, 'include')]

# mimic tensorflow cuda include setup so that their include command work
if not os.path.exists(os.path.join(root_path, "include")):
    os.mkdir(os.path.join(root_path, "include"))

cuda_inc_path = os.path.join(root_path, "include/cuda")
if not os.path.exists(cuda_inc_path) or os.readlink(cuda_inc_path) != CUDA_HOME:
    if os.path.exists(cuda_inc_path):
        os.remove(cuda_inc_path)
    os.symlink(CUDA_HOME, cuda_inc_path)
include_dirs += [os.path.join(root_path, 'include')]

# Ensure that all expected files and directories exist.
for loc in include_dirs:
    if not os.path.exists(loc):
        print(("Could not find file or directory {}.\n"
               "Check your environment variables and paths?").format(loc),
              file=sys.stderr)
        sys.exit(1)

lib_srcs = ['src/transducer_op_kernel.cc']

ext = setuptools.Extension('transducer_tensorflow.kernels',
                           sources=lib_srcs,
                           language='c++',
                           include_dirs=include_dirs,
                           library_dirs=[core_dir],
                           runtime_library_dirs=[core_dir],
                           libraries=['transducer_core',
                                      'tensorflow_framework'],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args)


class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

# Read the README.md file for the long description. This lets us avoid
# duplicating the package description in multiple places in the source.
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
with open(README_PATH, "r") as handle:
    # Extract everything between the first set of ## headlines
    LONG_DESCRIPTION = re.search(
        "#.*([^#]*)##", handle.read()).group(1).strip()

setuptools.setup(
    name="transducer_tensorflow",
    version="0.1",
    description="TensorFlow wrapper for transducer",
    license="Apache",
    author = "Lucky Wong",
    packages=["transducer_tensorflow"],
    ext_modules=[ext],
    cmdclass={'build_ext': build_tf_ext},
)
