"""
This sub-package implements the different methods that can be used to interpolate stellar spectra
to a new wavelength grid.

"""

from .GPmodel import GPSpecModel
from .scipy_interpol import ScipyInterpolSpecModel
