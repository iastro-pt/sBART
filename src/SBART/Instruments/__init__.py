"""
Data loading interfaces for data from different spectrographs. Each spectrograph is represented as a different class,
with all providing the same user-interface (after they load all the necessary information from disk).

.. note::
    During the years, the supported spectrographs have  been (or will be)  subjected to technical interventions, leading to
    changes in the instrumental profile. If SBART creates a stellar template that mixes information from
    before and after (such interventions), we will be adding noise into the model (as the spectral shape will be different)

    To avoid such issues, we introduce the concept of "subInstruments", which define individual data-sets within a
    spectrograph's lifespan. Thus, for each sub-Instrument, we create individual templates and extract independent
    radial velocities. For more details on the dates used to delimit the data-sets, we refer to the docs of the individual
    instruments.

"""

__all__ = [
    "ESPRESSO",
    "HARPS",
]
from .ESPRESSO import ESPRESSO
from .HARPS import HARPS


instrument_dict = {
    "ESPRESSO": ESPRESSO,
    "HARPS": HARPS,
}
