import numpy as np
import scipy.optimize as opt


def RVerror(rv, ccf, eccf=1.0):
    """
    Calculate the uncertainty on the radial velocity, following the same steps
    as the ESPRESSO DRS pipeline.

    Parameters
    ----------
    rv : array
    The velocity values where the CCF is defined.
    ccf : array
    The values of the CCF profile.
    eccf : array
    The errors on each value of the CCF profile.
    """

    ccf_slope = np.gradient(ccf, rv)
    ccf_sum = np.sum((ccf_slope / eccf) ** 2)
    return 1.0 / np.sqrt(ccf_sum)


def fgauss(x, a, mu, sigma, c, d):
    return c + a * np.exp(-((x - mu) ** 2.0) / (2.0 * sigma**2.0)) + d * np.array(x)


def ccffitRV(rv, ccf, eccf):
    mean = rv[np.argmin(ccf)]
    sigma = np.sqrt(sum((rv - mean) ** 2.0) / (len(rv)))
    guess = [max(ccf), mean, sigma, max(ccf), (ccf[-1] - ccf[0]) / (rv[-1] - rv[0])]
    # print(guess)
    fitpar, fitpar_cov = opt.curve_fit(fgauss, rv, ccf, guess, sigma=eccf, maxfev=10000)
    fitpar_err = np.sqrt(np.diag(fitpar_cov))
    fittedRV = fitpar[1]
    fittedRVerror = RVerror(rv, ccf, eccf)
    return (
        fittedRV,
        fittedRVerror,
        [fitpar[0], fitpar[1], abs(fitpar[2]), fitpar[3], fitpar[4]],
    )
