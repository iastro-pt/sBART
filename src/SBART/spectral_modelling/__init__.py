"""
This sub-package implements the different methods that can be used to interpolate stellar spectra
to a new wavelength grid.

"""
from .scipy_interpol import ScipyInterpolSpecModel
from .GPmodel import GPSpecModel
