def evaluate_polynomial(params, XX):
    """Generate a N-degree polynomial from the parameters present in the param argument.
    The constant term of the polynomial is set at zero!


    Parameters
    ----------
    params : list
        List of the parameters to construct the polynomial. The polynomial is of the format: \sum_{i=0}^{N} p_i * XX^i
    XX : np.ndarray
        POints in which the polynomial will be evaluated

    Returns
    -------
    np.ndarray
        Results from the evaluation of the polynomial
    """

    out = 0
    for coef_index, coef in enumerate(params):
        out += coef * XX ** (coef_index + 1)
    return out
