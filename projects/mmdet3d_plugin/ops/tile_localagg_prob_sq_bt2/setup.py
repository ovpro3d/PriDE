# Build the aggregation and shared/own-CE backtrace extension.
# Run `python setup.py build_ext --inplace` from this directory.
#
# Based on the Inria GRAPHDECO local aggregation code (original headers kept in src/);
# free for non-commercial, research and evaluation use under LICENSE.md.
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="tile_local_aggregate_prob_sq_bt2",
    packages=['tile_local_aggregate_prob_sq_bt2'],
    ext_modules=[
        CUDAExtension(
            name="tile_local_aggregate_prob_sq_bt2._C",
            sources=[
            "src/aggregator_impl.cu",
            "src/forward.cu",
            "src/backward.cu",
            "src/backtrace.cu",
            "src/backtrace_own.cu",
            "local_aggregate.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
