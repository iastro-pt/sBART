import time
import traceback
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
import tqdm
from loguru import logger
from tabletexifier import Table

from SBART.Masks import Mask
from SBART.utils import custom_exceptions, open_buffer
from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks
from SBART.utils.UserConfigs import (
    DefaultValues,
    UserParam,
    ValueFromList,
)
from SBART.utils.concurrent_tools.close_interfaces import close_buffers, kill_workers
from SBART.utils.custom_exceptions import (
    BadOrderError,
    BadTemplateError,
    InvalidConfiguration,
)
from SBART.utils.shift_spectra import interpolate_data, remove_RVshift
from SBART.utils.status_codes import INTERNAL_ERROR
from SBART.utils.units import kilometer_second
from .Stellar_Template import StellarTemplate


class SumStellar(StellarTemplate):
    """
    This is the usual stellar template, that is constructed by shifting all observations to a common, rest,
    frame (i.e. by removing their stellar RVs) and computing a mean of all observations. Uncertainties can be
    propagated either through an analytical formula or by interpolating the original uncertainties of each
    observation.

    **User parameters:**

        This object doesn't introduce unique user parameters.

    *Note:* Check the **User parameters** of the parent classes for further customization options of this class

    """

    method_name = "Sum"
    _default_params = StellarTemplate._default_params + DefaultValues(
        ALIGNEMENT_RV_SOURCE=UserParam("DRS", constraint=ValueFromList(["DRS", "SBART"])),
    )

    def __init__(self, subInst: str, user_configs: Optional[dict] = None, loaded: bool = False):
        super().__init__(subInst=subInst, user_configs=user_configs, loaded=loaded)

        if not loaded:
            if self._internal_configs["MEMORY_SAVE_MODE"]:
                logger.warning(
                    "Stellar template creation will save RAM usage. This will result in multiple open/close "
                    "operations across the entire SBART pipeline! "
                )

            self._found_error = False

    @custom_exceptions.ensure_invalid_template
    def create_stellar_template(self, dataClass, conditions=None) -> None:
        """
        Creating the stellar template
        """
        # removal may change the first common wavelength; make sure
        try:
            super().create_stellar_template(dataClass, conditions)
        except custom_exceptions.StopComputationError:
            return

        if self._internal_configs["ALIGNEMENT_RV_SOURCE"] == "SBART":
            try:
                dataClass.load_previous_SBART_results(
                    self._internal_configs["PREVIOUS_SBART_PATH"],
                    use_merged_cube=self._internal_configs["USE_MERGED_RVS"],
                )

            except custom_exceptions.InvalidConfiguration:
                self.add_to_status(INTERNAL_ERROR)
                logger.critical("SBART RV loading routine failed. Stopping template creation")
                return
        instrument_information = dataClass.get_instrument_information()
        epoch_shape = instrument_information["array_size"]
        # Create arrays of zeros in order to open in shared memory and change their values!
        self.spectra = np.zeros(epoch_shape)
        self.rejection_array = np.zeros((len(self.frameIDs_to_use), epoch_shape[0]))

        self.wavelengths = np.zeros(epoch_shape)
        self.uncertainties = np.zeros(epoch_shape)
        self.spectral_mask = None

        try:
            self.package_pool = Queue()
            self.output_pool = Queue()

            self.launch_parallel(dataClass)
            self.evaluate_bad_orders()
            self._finish_template_creation()

        except Exception as e:
            logger.opt(exception=True).critical("Stellar template creation failed due to: {}", e)
        finally:
            logger.info("Closing shared memory interfaces of the Stellar template")
            self.cleanup_shared_memory()

        if 0:  # FOR now disabled, low PRIO
            snrs = np.zeros((len(self.used_epochs), dataClass.instrument.N_orders))
            for index, epo in enumerate(self.used_epochs):
                snrs[index] = dataClass.instrument.snrs[epo]

            # expected template performance:
            median = np.median(snrs, axis=0)
            worst = np.min(snrs, axis=0)
            best = np.max(snrs, axis=0)
            table = Table(
                [
                    "Order",
                    "Median",
                    "Best SNR",
                    "Worst SNR",
                    "Worst epoch",
                    "TEMPLATE (median SNR)",
                ]
            )
            table.set_decimal_places(2)

            for order in range(dataClass.instrument.N_orders):
                if order in self.bad_orders:
                    continue

                mask = ~self.mask[order]
                snr = self.spectra[order][mask] / self.uncertainties[order][mask]

                table.add_row(
                    [
                        str(order),
                        median[order],
                        best[order],
                        worst[order],
                        np.argmin(snrs, axis=0)[order],
                        np.median(snr),
                    ]
                )

            logger.info(f"SNR analysis:{table}")
        else:
            logger.warning("Computation of SNR from stellar template temporarily disabled!")

        if 0:  # compute_statistics:
            self._mask.compute_statistics()

    def launch_parallel(self, dataClass):

        inst_info = dataClass.get_instrument_information()
        N_orders = inst_info["array_size"][0]

        epoch_errors = dataClass.collect_RV_information(
            KW=self.RV_keyword + "_ERR",
            subInst=self._associated_subInst,
            frameIDs=self.frameIDs_to_use,
            units=kilometer_second,
            as_value=True,
            include_invalid=False,
        )

        epochsRVs = dataClass.collect_RV_information(
            KW=self.RV_keyword,
            frameIDs=self.frameIDs_to_use,
            subInst=self._associated_subInst,
            units=kilometer_second,
            as_value=True,
            include_invalid=False,
        )

        chosen_epochID = self.frameIDs_to_use[np.argmin(epoch_errors)]

        wave_reference, _, _, _ = dataClass.get_frame_arrays_by_ID(chosen_epochID)

        self.wavelengths = remove_RVshift(
            wave_reference,
            stellar_RV=epochsRVs[np.argmin(epoch_errors)],
        )

        logger.info(
            "Using observation from {} as a basis for stellar template construction",
            dataClass.get_frame_by_ID(chosen_epochID),
        )

        logger.info(
            "Propagation spectral uncertainties through: {}",
            self._internal_configs["INTERPOLATION_ERR_PROP"],
        )
        logger.info("Using frameIDs: {}", self.frameIDs_to_use)

        kwargs = {
            "valid_epochIDs": self.frameIDs_to_use,
            "chosen_epochID": chosen_epochID,
            "subInst": self._associated_subInst,
            "N_orders": N_orders,
            "interpol_cores": self._internal_configs["NUMBER_WORKERS"][1],
            "interpol_prop_type": self._internal_configs["INTERPOLATION_ERR_PROP"],
            "dataClass": dataClass,
        }

        logger.info("Lauching {} workers!", self._internal_configs["NUMBER_WORKERS"][0])
        logger.info(
            "Using {} cores for each interpolation!", self._internal_configs["NUMBER_WORKERS"][1]
        )

        # TODO: Avoid error ir we launch this after the template is already in shared memory!
        shr_wave, shr_tmp, shr_uncert, shr_counts = self.convert_to_shared_mem()
        buffers = self.shm

        for _ in range(self._internal_configs["NUMBER_WORKERS"][0]):
            _ = tqdm.tqdm(
                total=len(self.frameIDs_to_use) // self._internal_configs["NUMBER_WORKERS"][0],
                leave=False,
            )
            p = Process(
                target=self.perform_calculations,
                args=(self.package_pool, self.output_pool, buffers),
                kwargs=kwargs,
            )
            p.start()

        RunTimeRejections = []

        for frameID in self.frameIDs_to_use:
            # to avoid multiple processes opening the arrays at the same time, we open it beforehand
            logger.info("Starting frameID: {}", frameID)
            try:
                _ = dataClass.load_frame_by_ID(frameID)
            except custom_exceptions.FrameError:
                logger.warning("Run Time rejection of frameID = {}", frameID)
                RunTimeRejections.append(frameID)
                continue

            self.used_fpaths.append(dataClass.get_filename_from_frameID(frameID, full_path=True))

            total_number_packages = 0
            for order in range(N_orders):
                self.package_pool.put((frameID, order))
                total_number_packages += 1
            t = time.time()
            received = 0

            while received != total_number_packages:
                comm_out = self.output_pool.get()
                if not isinstance(comm_out, tuple) and not np.isfinite(comm_out):
                    logger.critical("non finite output")
                    kill_workers([], self.package_pool, self._internal_configs["NUMBER_WORKERS"][0])
                    self._found_error = True
                    raise BadTemplateError("Template creation failed")

                frameID, order, rejection = comm_out
                self.rejection_array[self.frameIDs_to_use.index(frameID), order] = rejection
                received += 1
            logger.debug(f"Frame took {time.time() - t :0f} seconds")

            if self._internal_configs["MEMORY_SAVE_MODE"]:
                _ = dataClass.close_frame_by_ID(frameID)

        # account for possible Frame Rejections after opening the fits data units
        valid_inds = []
        for frameID in self.frameIDs_to_use:
            if frameID not in RunTimeRejections:
                valid_inds.append(self.frameIDs_to_use.index(frameID))
        self.rejection_array = self.rejection_array[valid_inds]

        for frameID in RunTimeRejections:
            self.frameIDs_to_use.remove(frameID)

        logger.info("Updating template mask")

        self.spectra = shr_tmp[:] / len(self.frameIDs_to_use)

        new_mask = np.zeros(self.spectra.shape, dtype=np.bool)

        # plt.plot(shr_counts[2], marker = 'x', linestyle ='')
        # plt.axhline(len(self.frameIDs_to_use))
        # plt.show()
        new_mask[np.where(shr_counts != len(self.frameIDs_to_use))] = True
        new_mask[np.where(self.spectra < 1)] = True

        logger.debug("Ensuring increasing wavelenghs in the stellar template")
        # ENsure that we always have increasing wavelengths
        # By construction, the wavelength solution of the template is the "un-masked" wavelengths of the chosen frame
        # When opening the frames we search for non-increasing wavelengths. However, that "mask" is not translated into the
        # actual wavelength solution of the template. The goal of this is to reject those regions (this does not have any
        # impact on the non-affected regions).
        diffs = np.where(np.diff(self.wavelengths, axis=1) < 0)
        if diffs[0].size > 0:
            new_mask[diffs] = True

        self.spectral_mask = Mask(new_mask, mask_type="binary")
        # error propagation for the mean
        self.uncertainties = np.sqrt(shr_uncert[:]) / len(self.frameIDs_to_use)

    def perform_calculations(self, in_queue, out_queue, buffer_info, **kwargs):
        """
        Compute the stellar template from the input S2D data. Accesses the data from shared memory arrays!
        """

        chosen_epoch = kwargs["chosen_epochID"]
        current_subInst = kwargs["subInst"]
        used_epochs = kwargs["valid_epochIDs"]
        interpol_cores = kwargs["interpol_cores"]
        DataClassProxy = kwargs["dataClass"]

        shared_buffers = []
        (
            stellar_template,
            stellar_template_errors,
            stellar_template_wavelengths,
            counts,
            shared_buffers,
        ) = open_buffer(buffer_info, open_type="template", buffers=shared_buffers)

        pixels_in_order = stellar_template[0].size
        try:
            while True:

                data_in = in_queue.get()
                continue_computation = True
                if not isinstance(data_in, (list, tuple)):
                    if not np.isfinite(data_in):
                        return
                    else:
                        logger.critical("Wrong data format in the communication queue")
                        raise InvalidConfiguration

                frameID, order = data_in

                try:
                    (
                        wavelengths,
                        s2d_data,
                        s2d_uncerts,
                        s2d_mask,
                    ) = DataClassProxy.get_frame_OBS_order(frameID, order)
                except BadOrderError:
                    continue_computation = False

                if continue_computation:
                    current_epochRV = DataClassProxy.collect_RV_information(
                        KW=self.RV_keyword,
                        subInst=current_subInst,
                        frameIDs=[frameID],
                        units=kilometer_second,
                        as_value=True,
                    )[0]

                    wavelengths_to_interpolate = np.zeros(
                        stellar_template_wavelengths[order].shape, dtype=np.bool
                    )

                    # until now the mask has ones in the regions to remove
                    blocks = build_blocks(np.where(~s2d_mask))
                    shifted_wavelengths = remove_RVshift(wavelengths, current_epochRV)

                    for block in blocks:
                        start = remove_RVshift(wavelengths[block[0]], current_epochRV)
                        end = remove_RVshift(wavelengths[block[-1]], current_epochRV)
                        interpolation_indexes = np.where(
                            np.logical_and(
                                stellar_template_wavelengths[order] >= start,
                                stellar_template_wavelengths[order] <= end,
                            )
                        )
                        wavelengths_to_interpolate[interpolation_indexes] = True

                    template_indices = wavelengths_to_interpolate
                    spectral_mask = ~s2d_mask

                    try:
                        interpolated_order, interpolated_errors, _ = interpolate_data(
                            original_lambda=shifted_wavelengths[spectral_mask],
                            original_spectrum=s2d_data[spectral_mask],
                            original_errors=s2d_uncerts[spectral_mask],
                            new_lambda=stellar_template_wavelengths[order][template_indices],
                            lower_limit=0,
                            upper_limit=np.inf,
                            propagate_interpol_errors=kwargs["interpol_prop_type"],
                            interpol_cores=interpol_cores,
                        )
                    except Exception as e:
                        logger.critical("Interpolation failed due to: {}", e)
                        raise e

                    stellar_template[order][wavelengths_to_interpolate] += interpolated_order
                    stellar_template_errors[order][wavelengths_to_interpolate] += (
                        interpolated_errors ** 2
                    )
                    a = counts[order]
                    a[wavelengths_to_interpolate] = a[wavelengths_to_interpolate] + 1
                    counts[order] = a

                    valid_pixels = np.sum(wavelengths_to_interpolate)
                else:
                    valid_pixels = 0
                out_queue.put((frameID, order, (pixels_in_order - valid_pixels) / pixels_in_order))
        except Exception as e:
            # TODO: fix the procedure for when the workers die

            print(traceback.print_tb(e.__traceback__))
            close_buffers(shared_buffers)

            out_queue.put(np.inf)
            print(f"Template creation dead due to {e}, in order {order}")
            return INTERNAL_ERROR

    @property
    def RV_keyword(self) -> str:
        if self._internal_configs["ALIGNEMENT_RV_SOURCE"] == "SBART":
            RV_KW_start = "previous_SBART_RV"
        else:
            RV_KW_start = "DRS_RV"

        return RV_KW_start
