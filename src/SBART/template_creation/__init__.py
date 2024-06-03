"""
Controls the creation of both stellar and telluric templates.
"""

from .Loaded_Template import LoadedTemplate
from .stellar_templates.sum_stellar import SumStellar
from .telluric_templates.telluric_from_tapas import TapasTelluric
from .telluric_templates.telluric_from_telfit import TelfitTelluric
