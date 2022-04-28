"""
.. _SamplerInit:

Provide, to the RV model, new radial velocities.

The samplers are implemented with a uniform interface (this can be assumed to be true unless it is
explicitly said that it is not!). Thus, if a given user input has a similar function in all samplers, it will
follow a common "logic":

- *rv_step* - RV_measurement :
        Most often used for numerical derivatives. The individual samplers provide more details.

- *rv_window* - Tuple[RV_measurement, RV_measurement]:
        Will be used to define a RV window (for each observation) inside which the sampler will have to select its next tentative
        RV value. This is assumed to be a "RV distance" away from the previous RV value that SBART loaded (either CCF or previous
        application of SBART).They must both be positive values, as we create a window
        of [RV_prev - rv_prior[0], RV_prev + rv_prior[1]].

"""

from .chi_squared_sampler import chi_squared_sampler
from .Laplace_approx import Laplace_approx
from .MCMC_sampler import MCMC_sampler
