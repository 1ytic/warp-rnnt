import io
import os
import re

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_compile_archs(nvcc_flags=None):
    """Get the target CUDA architectures from CUDA_ARCH_LIST env variable"""
    if nvcc_flags is None:
        nvcc_flags = []
    CUDA_ARCH_LIST = os.getenv("CUDA_ARCH_LIST", None)
    if CUDA_ARCH_LIST is not None:
        for arch in CUDA_ARCH_LIST.split(";"):
            m = re.match(r"^([0-9.]+)(?:\(([0-9.]+)\))?(\+PTX)?$", arch)
            assert m, "Wrong architecture list: %s" % CUDA_ARCH_LIST
            com_arch = m.group(1).replace(".", "")
            cod_arch = m.group(2).replace(".", "") if m.group(2) else com_arch
            ptx = True if m.group(3) else False
            nvcc_flags.extend(
                ["-gencode", "arch=compute_{},code=sm_{}".format(com_arch, cod_arch)]
            )
            if ptx:
                nvcc_flags.extend(
                    [
                        "-gencode",
                        "arch=compute_{},code=compute_{}".format(com_arch, cod_arch),
                    ]
                )
    return nvcc_flags


def get_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with io.open(req_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


if not torch.cuda.is_available():
    raise Exception("CPU version is not implemented")

extra_compile_args = {
    "cxx": ["-std=c++11", "-O3", "-fopenmp"],
    "nvcc": ["-std=c++11", "-O3", "--compiler-options=-fopenmp"],
}

CC = os.getenv("CC", None)
if CC is not None:
    extra_compile_args["nvcc"].append("-ccbin=" + CC)

extra_compile_args["nvcc"].extend(get_cuda_compile_archs())

sources = ["binding.cpp", "../core.cu"]

requirements = get_requirements()

setup(
    name="warp_rnnt",
    version="0.0.1",
    description="PyTorch bindings for CUDA-Warp RNN-Transducer",
    url="https://github.com/1ytic/warp-rnnt",
    author="Ivan Sorokin",
    author_email="sorokin.ivan@inbox.ru",
    license="MIT",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="warp_rnnt._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    setup_requires=requirements,
    install_requires=requirements,
)
