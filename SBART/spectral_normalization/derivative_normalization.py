import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

from SBART.utils import custom_exceptions
from loguru import logger

from SBART.spectral_normalization.normalization_base import NormalizationBase
from SBART.utils.UserConfigs import (
    DefaultValues
)
from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks

class Derivative_normalization(NormalizationBase):
    """

    **User parameters:**


    *Note:* Also check the **User parameters** of the parent classes for further customization options of SBART

    """

    _default_params = NormalizationBase._default_params + DefaultValues()
    _name = "derivative"
    orderwise_application = True

    def __init__(self, obj_info, user_configs):
        super().__init__(obj_info=obj_info,
                         user_configs=user_configs,
                         )

    def _fit_orderwise_normalization(self, wavelengths, flux, uncertainties):
        print("here")
        if wavelengths[1000] > 7000:
            deriva = []
            error_deriva = []
            steps = np.diff(wavelengths)
            for index in range(1, flux.size-1):
                step_ratio = steps[index] / steps[index-1]
                deriva.append((flux[index + 1] - flux[index-1]*step_ratio**2 - (1 - step_ratio**2)*flux[index])/(steps[index]*(1 + step_ratio)))
                error_deriva.append(np.sqrt(uncertainties[index+1]**2 + (uncertainties[index-1]*step_ratio**2)**2 + (1-uncertainties[index]*step_ratio**2)**2)/(steps[index] * (1+step_ratio)))

            deriva = np.asarray(deriva)

            fig, axis = plt.subplots(3,1, sharex=True)
            axis[1].errorbar(wavelengths[1:-1], deriva, error_deriva)
            filter = median_filter(deriva, 1000)
            axis[1].plot(wavelengths[1:-1], filter, color = "red", ls="--")
            second_deriva = []
            error_second_deriva = []
            for index in range(1, np.asarray(deriva).size-1):
                step_ratio = steps[index] / steps[index-1]
                second_deriva.append((deriva[index + 1] - deriva[index-1]*step_ratio**2 - (1 - step_ratio**2)*deriva[index])/(steps[index]*(1 + step_ratio)))

                error_second_deriva.append(np.sqrt(error_deriva[index+1]**2 + (error_deriva[index-1]*step_ratio**2)**2 + (1-error_deriva[index]*step_ratio**2)**2)/(steps[index] * (1+step_ratio)))
            second_deriva = np.asarray(second_deriva)
            axis[2].errorbar(wavelengths[2:-2], second_deriva, error_second_deriva)
            second_filter = median_filter(second_deriva, 1000)
            axis[2].plot(wavelengths[2:-2], second_filter, color = "red", ls="--")
            axis[0].scatter(wavelengths, flux)
            derivative_regions = np.where(np.logical_or(np.subtract(deriva, 2*np.asarray(error_deriva)) >= 0,
                                                        np.subtract(deriva, -2*np.asarray(error_deriva)) <= 0))

            axis[1].scatter(wavelengths[1:-1][derivative_regions], deriva[derivative_regions], color="red", marker = "x")

            # axis[0].scatter(wavelengths[1:-1][derivative_regions], flux[1:-1][derivative_regions], color = "red", ls="--")
            second_derivative_regions = np.where(np.logical_or(np.subtract(second_deriva, 2*np.asarray(error_second_deriva)) >= 0,
                                                        np.subtract(second_deriva, -2*np.asarray(error_second_deriva)) <= 0))
            axis[2].scatter(wavelengths[2:-2][second_derivative_regions], second_deriva[second_derivative_regions], color="red", marker = "x")

            blocks = build_blocks(derivative_regions)
            print(blocks)
            pixel_jumps = []
            for b_index in range(len(blocks) -1):
                pixel_jumps.append(blocks[b_index+1][0] - blocks[b_index][-1])
            print(pixel_jumps)
            marked_regions = []
            new_block = True
            for jump_index, jump in enumerate(pixel_jumps): # TODO: missing the last block if it is not merged!
                if new_block:
                    start = blocks[jump_index][0]
                    end = blocks[jump_index][-1]

                if jump < 10:
                    end = blocks[jump_index+1][-1]
                    new_block = False
                else:
                    new_block=True
                    marked_regions.extend(range(start, end+1))
            if marked_regions[-1] != blocks[-1][-1]:
                marked_regions.extend(range(blocks[-1][0], blocks[-1][-1]+1))
            print(marked_regions)
            axis[0].scatter(wavelengths[1:-1][marked_regions], flux[1:-1][marked_regions], color = "red", ls="--", marker='d')
            clean = [i for i in range(9109) if i not in marked_regions]
            # axis[2].scatter(wavelengths[1:-1][clean], flux[1:-1][clean], color = "black", ls="--", marker='d')
            axis[0].set_ylabel("Spectra")
            axis[1].set_ylabel("1st Derivative")
            axis[2].set_ylabel("2nd Derivative")
            axis[2].set_xlabel("Wavelength")
            axis[0].set_title("Tau Ceti data")
            plt.show()
        return *self.apply_normalization(wavelengths, flux, uncertainties), {"ddd":21,
                                                                             "kasdkhjjkasdha":1
                                                                             }

    def _apply_orderwise_normalization(self, wavelengths, flux, uncertainties, **kwargs):
        super()._apply_orderwise_normalization(wavelengths, flux, uncertainties, **kwargs)
        return flux/10, uncertainties/10
    def _normalization_sanity_checks(self):
        super()._normalization_sanity_checks()
        # TODO: see what kind of data we want to use!
