from setuptools import setup, Extension
import os
import platform
import numpy

include_dirs = [
    os.path.abspath("include"),  # Path to `uhash.h`
    numpy.get_include(),         # NumPy include directory
]

# Set compiler and linker flags for OpenMP when using Conda
if platform.system() == "Darwin":  # macOS
    conda_prefix = os.environ.get("CONDA_PREFIX", "/usr/local")
    if os.path.isdir(conda_prefix):
        extra_compile_args = ["-Xpreprocessor", "-fopenmp"]
        extra_link_args = ["-lomp"]
        include_dirs.append(os.path.join(conda_prefix, "include"))
        library_dirs = [os.path.join(conda_prefix, "lib")]
    else:
        print(f"Warning: CONDA_PREFIX path '{conda_prefix}' is invalid or does not exist. Disable OpenMP support on macOS.")
        extra_compile_args = ["-Wall", "-Wextra"]
        extra_link_args = []
        library_dirs = []
else:  # Other systems
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-lgomp"]
    library_dirs = []

# Define the extension module
extension = Extension(
    "subg.subg",  # Package name
    sources=["src/subg/subg_acc.c"],  # Source file(s)
    include_dirs=include_dirs,  # Include directories
    extra_compile_args=extra_compile_args,  # Compiler flags
    extra_link_args=extra_link_args,        # Linker flags
)

setup(
    name="subg",
    version="2.3.0",
    description="An C and OpenMP extension for Python on accelerating subgraph operations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Harris Yin",
    author_email="yinht@purdue.edu",
    url="https://github.com/VeritasYin/subg_acc",
    package_dir={"": "src"},
    packages=["subg"],
    ext_modules=[extension],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "License :: BSD 2-Clause",
        "Operating System :: Unix",
    ],
    python_requires=">=3.7",
    install_requires=["numpy"],
)

