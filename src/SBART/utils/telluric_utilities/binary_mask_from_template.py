import numpy as np
from loguru import logger


def create_binary_template(transmittance, continuum_level, percentage_drop) -> None:
    """Transform the transmittance spectra into a binary template,
    based on percentage deviations from the continuum level.

    Parameters
    ----------
    continuum_level : np.ndarray()
        Continuum level that is estimated by the children classes

    """
    logger.info("Converting from transmittance spectra to binary mask!")

    # Find a decrease of 1% in relation to the continuum level; Positive
    # gains (against the continuum value) are not considered as tellurics
    percentages = (transmittance - continuum_level) / continuum_level
    telluric_indexes = np.where(percentages < -percentage_drop / 100)

    # We want a binary template
    telluric_mask = np.zeros_like(transmittance, dtype=int)
    telluric_mask[telluric_indexes] = 1

    return telluric_mask
