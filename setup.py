from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

VERSION = "0.1.0"
BUILD_NATIVE_CUDA = os.environ.get("CUMINE_BUILD_CUDA", "").strip() == "1"


def find_cuda_home() -> Path | None:
    env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent

    p = Path("/usr/local/cuda")
    if p.exists():
        return p

    return None


class BuildCudaExtension(build_ext):
    def build_extensions(self):
        cuda_home = find_cuda_home()
        if cuda_home is None:
            raise RuntimeError(
                "CUMINE_BUILD_CUDA=1 was set, but CUDA_HOME/nvcc was not found. "
                "Install the CUDA toolkit or set CUDA_HOME."
            )

        self.compiler.src_extensions.append(".cu")
        original_compile = self.compiler._compile

        def compile_cuda_or_cpp(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                nvcc = os.environ.get("NVCC") or str(cuda_home / "bin" / "nvcc")
                include_args = []
                for inc in self.include_dirs:
                    include_args.extend(["-I", inc])

                cmd = [
                    nvcc,
                    "-c",
                    src,
                    "-o",
                    obj,
                    "-O3",
                    "-std=c++14",
                    "--compiler-options",
                    "-fPIC",
                ] + include_args

                arch = os.environ.get("CUMINE_CUDA_ARCH")
                if arch:
                    # Example: CUMINE_CUDA_ARCH=sm_86
                    cmd.extend(["-arch", arch])

                subprocess.check_call(cmd)
            else:
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = compile_cuda_or_cpp
        super().build_extensions()


ext_modules = []
cmdclass = {}

if BUILD_NATIVE_CUDA:
    cuda_home = find_cuda_home()
    if cuda_home is None:
        # BuildCudaExtension will raise a clearer error later, but keep paths sane.
        cuda_home = Path("/usr/local/cuda")

    ext_modules.append(
        Extension(
            "cumine._cuda_ext",
            sources=["cumine/cuda_ext.cpp", "cumine/cuda_kernels.cu"],
            include_dirs=[np.get_include(), str(cuda_home / "include")],
            library_dirs=[str(cuda_home / "lib64"), str(cuda_home / "lib")],
            libraries=["cudart"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++14"],
        )
    )
    cmdclass["build_ext"] = BuildCudaExtension


setup(
    name="cumine",
    version=VERSION,
    description="MINE-style dependency statistics with CPU, CuPy, and native CUDA backends",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.18"],
    extras_require={
        "gpu": ["cupy-cuda12x"],
        "cupy": ["cupy-cuda12x"],
        "cuda": [],
        "dev": ["pytest"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
