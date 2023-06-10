from typing import List

import numpy as np
from loguru import logger
from tabletexifier import Table

from SBART.utils.air_to_vac import airtovac
from SBART.utils.shift_spectra import apply_RVshift
from SBART.utils.units import kilometer_second


class Indicators:
    def __init__(self):
        """
        Features - air wavelength, < lower, higher > lower and higher in angstrom

        The feature list uses air wavelengths! When comparing with the spectra the
        indicator_window property automatically converts to vacuum wavelengths!!!!
        """
        # List of features in 'air' wavelengths
        self.air_features = {  # central wavelength from:
            "CaK": (3933.66, -0.3, 0.3),  # Robertson et al. (2016, ApJ)
            "CaH": (3968.47, -0.2, 0.2),  # Robertson et al. (2016, ApJ)
            "H_eps": (3970.075, -0.3, 0.3),  # balmer series (7 -> 2)
            "H_delta": (4101.734, -0.7, 0.7),  # balmer series (6 -> 2)
            "H_gamma": (4340.472, -1.0, 1.0),  # balmer series (5 -> 2)
            "H_beta": (4861.35, -0.9, 0.9),  # balmer series (4 -> 2)
            "NaID_1": (5889.96, -0.7, 0.7),  # Robertson et al. (2016, ApJ)
            "NaID_2": (5895.93, -0.45, 0.45),  # Robertson et al. (2016, ApJ)
            "H_alpha": (6562.808, -1.0, 1.0),  # Kuerster et al. (2003, A&A)
            "CaI": (6572.795, -0.9, 0.9),  # Kuerster et al. (2003, A&A)
            "K_I2": (7664.90, -0.5, 0.5),  # Robertson et al. (2016, ApJ)
        }

        self.vacuum_features = {}

        self._blocked_features = []

        self.marked_orders = {}

        self._removal_table = Table(["name", "Wavelengths (air)", "orders"])

    def add_feature(self, name, region, vacuum_wavelengths=False):
        """Add one extra "indicator" to be removed from the spectra

        Parameters
        ----------
        name : str
            Name of the feature. Must not exist in either the 'air' or 'vacuum' features
        region : tuple
            Region to be removed: (center of the region; left extension; right_extension). Evrything in angstrong
            The region can also be a (start, end) tuple. The center of the region will be the mean of the two edges.

        vacuum_wavelengths : bool, optional
            If True, the region wavelengths are in vacuum, by default False

        Raises
        ------
        Exception
            [description]
        Exception
            [description]
        Exception
            [description]
        """
        if name in self.air_features:
            raise Exception("Keyword already in use")
        if len(region) == 2:  # assume it's (start, end)
            c = np.mean(region)
            ptp = np.ptp(region)
            region = (c, -ptp / 2, ptp / 2)
        elif len(region) != 3:
            raise Exception("Wrong format for the window")

        if name in self.air_features or name in self.vacuum_features:
            raise Exception("Feature {} already exists".format(name))

        if vacuum_wavelengths:
            self.vacuum_features[name] = region

        else:
            self.air_features[name] = region

    def clear_all_features(self) -> None:
        """Remove all reasons to mask any wavelength region!"""
        logger.warning("Discarding all saved features from the activity indicators")
        self.air_features = {}
        self.vacuum_features = {}
        self._blocked_features = []

    def disabled_indicator(self, features: list) -> None:
        """
        Receive a list of names that will **not** be removed from the observations
        """
        self._blocked_features.extend(features)

    def disable_all(self):
        logger.warning("Disabling all indicators")
        self.disabled_indicator(list(self.air_features.keys()))

    def compute_forbidden_wavelengths(self, stellar_RV) -> List[tuple]:
        wavelength_blocks = []
        _, ind_windows = self.indicator_window
        for indicator_window in ind_windows:
            # The stellar spectra is already corrected from the BERV
            lower_bound = apply_RVshift(indicator_window[0], stellar_RV.to(kilometer_second).value)
            upper_bound = apply_RVshift(indicator_window[1], stellar_RV.to(kilometer_second).value)
            wavelength_blocks.append((lower_bound, upper_bound))

        return wavelength_blocks

    @property
    def indicator_window(self):
        out = []
        lines = []
        for feat_name, specs in self.air_features.items():
            tmp = []
            lines.append(feat_name)
            central_wave = airtovac(specs[0])
            for side in specs[1:]:
                tmp.append(central_wave + side)

            out.append(tmp)
        for feat_name, specs in self.vacuum_features.items():
            lines.append(feat_name)
            central_wave = specs[0]
            for side in specs[1:]:
                tmp.append(central_wave + side)

        if len(lines) == 0:
            logger.warning(
                "No activity-related features are enabled. Removing nothing from the spectra"
            )
        return lines, out

    def get_line_information(self, feature_name, vacuum_wavelength=False):
        if vacuum_wavelength:
            return self.vacuum_features[feature_name]
        return self.air_features[feature_name]

    @property
    def get_formatted_table(self):
        return self._removal_table.get_pretty_print(ignore_cols=[])
