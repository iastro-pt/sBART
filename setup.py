import os
from pathlib import Path

import numpy

import setuptools

curr_file = Path(__file__).parent.absolute()

from setuptools import Extension

version = "0.5.1"

USE_CYTHON = False  # command line option, try-import, ...

ext = ".pyx" if USE_CYTHON else ".c"

targ_path = curr_file / "src" / "SBART" / "utils" / "cython_codes"

pyx_files = list(targ_path.glob(f"**/*{ext}"))
targets = {}
for entry in pyx_files:
    parts = entry.relative_to(curr_file).parts
    parts = parts[1:-1] + (parts[-1].split(".")[0],)
    targets[".".join(parts)] = (entry.relative_to(curr_file)).as_posix()


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


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

    ext_modules = cythonize(
        ext_modules,
        compiler_directives=compiler_directives,
    )

from distutils.core import setup

all_packages = setuptools.find_packages(
    where="src",
    # include=["SBART.*"]
)

requ_path = Path(__file__).parent
with open(requ_path / "requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="SBART",
    package_dir={"": "src"},
    version=version,
    description="Python Distribution Utilities",
    packages=all_packages,
    include_package_data=True,
    ext_modules=ext_modules,
    install_requires=required,
    include_dirs=[numpy.get_include()],
    # package_data={"SBART": ["utils/tapas_downloader",
    #                         "resources/*",
    #                         "outside_tools/*",
    #                         "utils/cython_codes/cubic_interpolation/__init__.py"
    #                         ]
    #               }
)
