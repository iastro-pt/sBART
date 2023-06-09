from pathlib import Path

import setuptools

curr_file = Path(__file__).parent.absolute()

from setuptools import Extension

version = "0.4.0"


USE_CYTHON = False   # command line option, try-import, ...

ext = '.pyx' if USE_CYTHON else '.c'

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
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    from Cython.Build import cythonize

    ext_modules = cythonize(ext_modules,
                        compiler_directives=compiler_directives,
                        )

from distutils.core import setup

all_packages = setuptools.find_packages(where=".", include=["SBART", "SBART.*"])
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='SBART',
      version=version,
      description='Python Distribution Utilities',
      packages=all_packages,
      include_package_data=True,
      ext_modules=ext_modules,
      install_requires=required
      )
