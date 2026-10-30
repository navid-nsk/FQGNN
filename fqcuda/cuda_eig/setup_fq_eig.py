"""Standalone build for the Op-B eigenphase CUDA op (module: fq_eig_ops).

A standalone extension, built against torch 2.11. The .cu
includes kernels/common.cuh (which uses <torch/types.h>) to dodge the torch-2.11/MSVC C2872 in
compiled_autograd.h under nvcc. fp32 (production) + fp64 (gradcheck) only.

CRITICAL: NO --use_fast_math (corrupts the fp32 gradient via __sinf/__cosf), NO TF32.
Build with the CUDA-12.8-compatible 14.44 MSVC toolset (see README.md, "CUDA extensions")."""
import os
import platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
kernels_dir = os.path.join(this_dir, "kernels")
bindings_dir = os.path.join(this_dir, "bindings")

cxx_flags = (["/O2", "/std:c++17", "/Zc:__cplusplus"]
             if platform.system() == "Windows" else ["-O2", "-std=c++17"])

setup(
    name="fq_eig_ops",
    ext_modules=[
        CUDAExtension(
            name="fq_eig_ops",
            sources=[
                os.path.join(bindings_dir, "fq_eig_bindings.cpp"),
                os.path.join(kernels_dir, "fq_eig.cu"),
            ],
            include_dirs=[kernels_dir],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": [
                    "-O2",
                    # NO --use_fast_math (fast-math sincos breaks the fp32 gradient)
                    "-allow-unsupported-compiler",
                    "-gencode=arch=compute_{a},code=sm_{a}".format(a=os.environ.get("FQCUDA_ARCH", "120")),
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
