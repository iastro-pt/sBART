import numpy
from pathlib import Path

import numpy as np
import setuptools

curr_file = Path(__file__).parent.absolute()
from setuptools import Extension
from Cython.Build import cythonize

import Cython.Compiler.Options
from Cython.Build import cythonize
from setuptools import Extension, setup
from SBART import version

USE_CYTHON = False   # command line option, try-import, ...

ext = '.pyx' if USE_CYTHON else '.c'


Cython.Compiler.Options.annotate = True

targ_path = curr_file / "SBART" / "utils" / "cython_codes"

pyx_files = targ_path.glob(f"**/*{ext}")

targets = {}
for entry in pyx_files:
    parts = entry.relative_to(curr_file).parts
    parts = parts[:-1] + (parts[-1].split(".")[0],)

    targets[".".join(parts)] = (entry.relative_to(curr_file )).as_posix()

ext_modules = [
    Extension(
        key,
        [value],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
    for key, value in targets.items()
]

compiler_directives = {"language_level": 3, "embedsignature": True}

if USE_CYTHON:
    ext_modules = cythonize(ext_modules,
                        compiler_directives=compiler_directives,
                        )

from distutils.core import setup

all_packages = setuptools.find_packages(where="SBART", include=["**"])

setup(name='SBART',
      version=version,
      package_dir={"": "SBART"},
      description='Python Distribution Utilities',
      packages=all_packages,
      include_package_data=True,
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],

      )
