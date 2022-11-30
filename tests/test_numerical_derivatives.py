import numpy as np

from SBART.utils.math_tools.numerical_derivatives import first_numerical_derivative


def test_first_numerical_derivative():
    XX = np.arange(0, 20, 1)

    for m, b in [[0, 1], [1, 1]]:
        YY = XX * m + b
        computed_value, uncert = first_numerical_derivative(XX, YY, YY)
        assert np.allclose(computed_value, m)
        assert np.allclose(computed_value, np.gradient(YY, XX))

    # testing non-uniform step:
    XX = 1.1 ** np.arange(0, 20, 1)

    for m, b in [[0, 1], [10, 1]]:
        YY = XX * m + b
        computed_value, uncert = first_numerical_derivative(XX, YY, YY)
        assert np.allclose(computed_value, m)
        assert np.allclose(computed_value, np.gradient(YY, XX))

    # Testing slightly more complex expressions:
    for trig_func in [np.sin, np.cos, np.tan]:
        XX = np.arange(0, 2 * np.pi, np.pi / 10)
        YY = trig_func(XX)
        computed_value, uncert = first_numerical_derivative(XX, YY, YY)
        grad_result = np.gradient(YY, XX)
        assert np.allclose(computed_value, grad_result)
