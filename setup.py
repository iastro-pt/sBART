# Ensure numpy is available before defining extensions
import numpy
from setuptools import Extension, find_packages, setup

# Try to use Cython if available
try:
    from Cython.Build import cythonize

    use_cython = True
except ImportError:
    use_cython = False

ext = ".pyx" if use_cython else ".c"

ext_modules = [
    Extension(
        "SBART.utils.cython_codes.matmul.second_term",
        [f"src/SBART/utils/cython_codes/matmul/second_term{ext}"],
        include_dirs=[numpy.get_include()],
    )
]

if use_cython:
    ext_modules = cythonize(ext_modules)

setup(
    name="SBART",
    version="1.1.2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
)
