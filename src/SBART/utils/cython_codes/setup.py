if __name__ == "__main__":
    print("COMPILING CYTHON")
    import Cython.Compiler.Options
    from Cython.Build import cythonize
    from setuptools import Extension, setup

    Cython.Compiler.Options.annotate = True

    targets = {
        "cubic_interpolation.inversion.inverter": "cubic_interpolation/inversion/inverter.pyx",
        "cubic_interpolation.partial_derivative.partial_derivative": "cubic_interpolation/partial_derivative/partial_derivative.pyx",
        "cubic_interpolation.second_derivative.second_derivative": "cubic_interpolation/second_derivative/second_derivative.pyx",
        "matmul.second_term": "matmul/second_term.pyx",
    }

    ext_modules = [
        Extension(
            key,
            [value],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
        for key, value in targets.items()
    ]
    setup(
        ext_modules=cythonize(
            ext_modules, compiler_directives={"language_level": "3"}, annotate=True
        )
    )
