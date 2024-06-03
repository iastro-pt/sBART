import numpy
from setuptools import Extension
from Cython.Build import cythonize

targets = [
    "cubic_interpolation.inversion.inverter",
    "cubic_interpolation.partial_derivative.partial_derivative",
    "cubic_interpolation.second_derivative.second_derivative",
    "matmul.second_term",
]

ext_modules=[
        Extension(
            f"SBART.utils.cython_codes.{i}", [f"SBART/utils/cython_codes/{i.replace('.','/')}.pyx"],

        )
        for i in targets
    ]


compiler_directives = {"language_level": 3, "embedsignature": True}


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                ext_modules, compiler_directives=compiler_directives
            ),

            "include_dirs": [numpy.get_include()],
        }
    )
