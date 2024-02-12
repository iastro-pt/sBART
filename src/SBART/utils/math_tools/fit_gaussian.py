from typing import Tuple

import numpy as np
from astropy.units import Quantity

try:
    from iCCF import gaussfit

    MISSING_iCCF = False
except ImportError:
    MISSING_iCCF = True
from loguru import logger

from SBART.utils.status_codes import CONVERGENCE_FAIL, SUCCESS, Flag
from SBART.utils.types import unitless_data_vector


def fit_CCF_gaussian(
    RV_array: unitless_data_vector, CCF_profile: unitless_data_vector, RV_units: Quantity
) -> Tuple[Quantity, Quantity, Flag]:
    """
    Implements a Gaussian fit to the CCF profile.
    Parameters
    ----------
    RV_units: Quantity
        Astropy units of the RV_array values
    RV_array: np.ndarray
        RV array for which we compute the CCF
    CCF_profile: np.ndarray
        CCF profile

    Returns
    -------
    float : fitted RV
    float : fitted RV uncertainty
    """

    if MISSING_iCCF:
        raise Exception("iCCF optional dependency is not installed")

    fit_status = SUCCESS

    rv_array = np.asarray(RV_array)
    ccf_profile = np.asarray(CCF_profile)

    mean = rv_array[np.argmin(ccf_profile)]
    sigma = np.sqrt(sum(np.subtract(rv_array, mean) ** 2.0) / (len(rv_array)))

    guess = [-np.ptp(ccf_profile), mean, sigma, 0.5 * (ccf_profile[0] + ccf_profile[-1])]

    try:
        (_, output_rv, _, _), (_, output_err, _, _) = gaussfit(
            x=rv_array, y=ccf_profile, p0=guess, return_errors=True
        )
    except:
        logger.opt(exception=True).critical("CCF gaussian fit failed")
        fit_status = CONVERGENCE_FAIL
        output_rv = np.nan
        output_err = np.nan

    return output_rv * RV_units, output_err * RV_units, fit_status
