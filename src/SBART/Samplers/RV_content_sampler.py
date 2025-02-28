"""Provides previous SBART values to compute the expected RV precision on the "effective" wavelength regions
of the stellar template
"""

from typing import Tuple

from SBART.Base_Models.Sampler_Model import SamplerModel
from SBART.utils.choices import RV_EXTRACTION_MODE
from SBART.utils.status_codes import SUCCESS, Flag
from SBART.utils.units import kilometer_second
from SBART.utils.work_packages import Package


class RVcontent_sampler(SamplerModel):
    """The Chi-squared sampler implements a bounded minimization of a chi-squared curve.

    This metric is defined in the RV_step worker. After finding the optimal value, fit a parabola to estimate the
    true minimum value and the RV that would be associated with it. It also uses the curvature of the chi squared
    curve to estimate the RV uncertainty.

    """

    _name = "RVcontent"

    def __init__(self, rv_step, rv_prior):
        """Parameters
        ----------
        rv_step: RV_measurement
            Step to use when computing the numerical derivatives of the metric function (for the parabolic fit)
        rv_prior
            Specifies the "effective" RV window that the sampler can use. More details in :ref:`here <SamplerInit>`.

        """
        super().__init__(
            mode=RV_EXTRACTION_MODE.ORDER_WISE,
            RV_step=rv_step,
            RV_window=rv_prior,
        )

    def optimize_orderwise(self, target, target_kwargs: dict) -> Tuple[Package, Flag]:
        """Compute the RV for an entire order, followed by a parabolic fit to estimate
        uncertainty and better adjust chosen RV

        Parameters
        ----------
        target : [type]
            [description]
        target_kwargs : [type]
            Input arguments of the target function. Must contain the following:
                - dataClassProxy,
                - frameID
                - order

        Returns
        -------
        [type]
            [description]

        """
        # TODO: have this from the target_kwargs:
        # "current_order": current_order,
        # "current_frameID": current_epochID,

        out_pkg = Package(("RV", "RV_uncertainty", "quality", "pix_sum_in_template"))
        # TODO: send here previous SBART RV!
        init_guess = 0

        (
            predicted_velocity,
            predicted_error,
            quality,
            pix_sum_in_template,
        ) = self.apply_orderwise(init_guess, target, target_kwargs)

        for key, value in [
            ("RV", predicted_velocity * kilometer_second),
            ("RV_uncertainty", predicted_error * kilometer_second),
            ("quality", quality),
            ("pix_sum_in_template", pix_sum_in_template),
        ]:
            out_pkg[key] = value

        return out_pkg, SUCCESS
