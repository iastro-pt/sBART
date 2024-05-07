import traceback
from multiprocessing import Queue

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from SBART.Quality_Control.outlier_detection import compute_outliers

from SBART.data_objects import DataClass
from SBART.utils import find_wavelength_limits
from SBART.utils.concurrent_tools.open_buffers import open_buffer
from SBART.utils.custom_exceptions import BadOrderError, InvalidConfiguration, StopComputationError
from SBART.utils.status_codes import (
    HIGH_CONTAMINATION,
    INTERNAL_ERROR,
    MASSIVE_RV_PRIOR,
    SHUTDOWN,
    SUCCESS,
)
from SBART.utils.units import kilometer_second, meter_second
from SBART.utils.work_packages import WorkerOutput


def worker(
    dataClassProxy: DataClass,
    input_queue: Queue,
    out_queue: Queue,
    worker_configs: dict,
    sampler=None,
):
    if sampler.mode == "epoch-wise":
        # open shared memory array to store the computed mask
        # to avoid multiple re-computations of outliers!
        mask_cache, cached_orders, shared_buffers = open_buffer(
            sampler.shared_buffers, open_type="BayesianCache", buffers=[]
        )
    else:
        shared_buffers = {}

    sampler_mode = sampler.mode

    previous_subInst = ""
    cached_effective_wavelengths = False
    StellarTemplate = None

    previous_frameID = None

    # True if the worker closes down by outside "orders"
    natural_exit = False
    try:
        while True:
            data = input_queue.get()
            if data["shutdown"]:
                # Raises exception to trigger the closure of shared data from this side!
                natural_exit = True
                break
            current_epochID = data["frameID"]
            current_order = data["order"]

            if data["subInst"] != previous_subInst:
                # Change in the subInstrument means that we must ask for a "new" stellar template
                subInst = data["subInst"]
                StellarTemplate = dataClassProxy.get_stellar_template(subInst)
                target_function = data["target_function"]
                previous_subInst = data["subInst"]

            if sampler_mode == "epoch-wise":
                if current_epochID != previous_frameID:
                    cached_orders[:] = False
                    previous_frameID = current_epochID
                cached_effective_wavelengths = cached_orders[current_order]
            else:
                cached_effective_wavelengths = False

            output_package = WorkerOutput()
            output_package["frameID"] = current_epochID
            output_package["order"] = current_order

            obs_rv = dataClassProxy.get_KW_from_frameID(
                worker_configs["RV_keyword"], current_epochID
            )

            RVLowerBound, RVUpperBound = data["RVprior"]
            order_status = SUCCESS
            try:
                (
                    spec_wave,
                    spec_s2d,
                    spec_uncert,
                    spec_mask,
                ) = dataClassProxy.get_frame_OBS_order(current_epochID, current_order)
            except BadOrderError:
                order_status = HIGH_CONTAMINATION("Bad order from frame")
                output_package["Total_Flux_Order"] = 0

            if order_status == SUCCESS:
                # invert the binary masks for the current order!
                (
                    temp_wave,
                    temp,
                    temp_uncerts,
                    temp_mask,
                ) = StellarTemplate.get_data_from_spectral_order(current_order)

                current_order_mask = ~spec_mask
                template_order_mask = ~temp_mask

                if not cached_effective_wavelengths:
                    # compute the "effective" wavelength region for the optimization
                    current_order_mask, found, n_iter = compute_outliers(
                        obs_rv=obs_rv.to(kilometer_second).value,
                        spectra_wavelengths=spec_wave,
                        spectra=spec_s2d,
                        spectra_mask=current_order_mask,
                        StellarTemplate=StellarTemplate,
                        template_wavelengths=temp_wave,
                        template_mask=template_order_mask,
                        worker_configs=worker_configs,
                        spec_uncert=spec_uncert,
                        order=current_order,
                        # epoch = current_epochID
                    )

                    current_order_mask = find_wavelength_limits(
                        current_order_mask=current_order_mask,
                        spectra_order_waves=spec_wave,
                        template_order_mask=template_order_mask,
                        template_order_wavelengths=temp_wave,
                        lower_limit=RVLowerBound.to(kilometer_second).value,
                        upper_limit=RVUpperBound.to(kilometer_second).value,
                        min_block_size=worker_configs["min_block_size"],
                        # order = current_order
                    )

                    if sampler_mode == "epoch-wise":
                        mask_cache[current_order][:] = current_order_mask[:]
                        cached_orders[current_order] = True
                else:
                    # load the cached data (in shared memory!)
                    current_order_mask = mask_cache[current_order]

                if current_order in [35, 37, 40, 41, 42] and 0:
                    plt.title("Order: {}".format(current_order))
                    print(
                        len(np.where(~spec_mask == 1)[0]),
                        len(np.where(current_order_mask)[0]),
                    )
                    plt.plot(spec_wave, spec_s2d, color="red", alpha=0.3)
                    plt.plot(
                        spec_wave[current_order_mask],
                        spec_s2d[current_order_mask],
                        color="black",
                        marker="x",
                        linestyle="",
                    )
                    # plt.plot(temp_wave, temp, color = 'red')
                    # plt.gca().twinx().plot(temp_wave, template_order_mask)
                    plt.figure()
                    plt.plot(spec_wave, current_order_mask, color="red")
                    plt.plot(spec_wave, ~spec_mask, color="blue")
                    plt.show()

                output_package["Total_Flux_Order"] = np.sum(spec_wave[current_order_mask])

                if spec_wave[current_order_mask].size < worker_configs["min_block_size"]:
                    order_status = MASSIVE_RV_PRIOR
                elif spec_wave[current_order_mask].size < worker_configs["min_pixel_in_order"]:
                    order_status = HIGH_CONTAMINATION(
                        f"Less than pixels {worker_configs['min_pixel_in_order']} on order"
                    )
                else:
                    # Apply the sampler for this spectral order
                    target_kwargs = {
                        "template_wave": temp_wave[template_order_mask],
                        # "template": temp[template_order_mask],
                        # "template_uncerts": temp_uncerts[template_order_mask],
                        "StellarTemplate": StellarTemplate,
                        "spectra_wave": spec_wave[current_order_mask],
                        "spectra": spec_s2d[current_order_mask],
                        "squared_spectra_uncerts": spec_uncert[current_order_mask] ** 2,
                        "RV step": sampler.RV_step,
                        "interpol_prop_type": worker_configs[
                            "uncertainty_prop_type"
                        ],  # TODO: change these 2 lines!
                        "worker_configs": worker_configs,
                        "make_plot": current_order in [35, 41] and 0,
                        "current_order": current_order,
                        "current_frameID": current_epochID,
                        "Effective_RV_Limits": (
                            RVLowerBound.to(meter_second).value,
                            RVUpperBound.to(meter_second).value,
                        ),
                        "SAVE_DISK_SPACE": worker_configs["SAVE_DISK_SPACE"],
                    }

                    if worker_configs["remove_OBS_from_template"]:
                        # We need to pass the frame if we want to remove the OBs from the template
                        target_kwargs["frame"] = dataClassProxy.get_frame_by_ID(current_epochID)
                    else:
                        target_kwargs["frame"] = None

                    full_target_kwargs = {
                        **target_kwargs,
                        **data["target_specific_configs"],
                    }

                    if "ANALYSIS_TYPE" in full_target_kwargs:
                        full_target_kwargs["full_S2D_wavelengths"] = spec_wave
                        full_target_kwargs["full_S2D_mask"] = current_order_mask
                        full_target_kwargs["previous_RV_OBS"] = obs_rv.to(kilometer_second).value

                    if sampler_mode == "epoch-wise":
                        if full_target_kwargs.get("compute_metrics", False):
                            output_value, model_misspec = target_function(
                                data["model_parameters"], **full_target_kwargs
                            )
                            output_package["log_likelihood_from_order"] = output_value
                            output_package["FluxModel_misspec_from_order"] = model_misspec
                        else:
                            output_value = target_function(
                                data["model_parameters"], **full_target_kwargs
                            )
                            output_package["log_likelihood_from_order"] = output_value

                    elif sampler_mode == "order-wise":
                        optimization_output, order_status = None, SUCCESS
                        if sampler is None:
                            raise InvalidConfiguration(
                                "When performing order-wise optimization we must provide a sampler"
                            )

                        try:
                            (
                                optimization_output,
                                order_status,
                            ) = sampler.optimize_orderwise(target_function, full_target_kwargs)
                        except Exception as e:
                            print(traceback.print_tb(e.__traceback__))
                            print("------> ", current_epochID, current_order)
                            print("spectra", spec_s2d[current_order_mask])
                            print("template: ", temp[template_order_mask])
                            logger.critical(f"{full_target_kwargs}")
                            logger.critical(f"{list(full_target_kwargs.keys())}")
                            logger.critical(f"{spec_s2d[current_order_mask]}")
                            logger.critical(f"{temp[template_order_mask]}")

                            raise StopComputationError(
                                f"RV optimization failed on {current_epochID=}, {current_order=}"
                            ) from e
                            # plt.title("{} - {}".format(current_epochID, current_order))
                            # plt.plot(
                            #     spec_wave[current_order_mask],
                            #     spec_s2d[current_order_mask],
                            #     color="black",
                            # )
                            # plt.plot(
                            #     temp_wave[template_order_mask],
                            #     temp[template_order_mask],
                            #     color="red",
                            #     linestyle="",
                            #     marker="x",
                            # )
                            # plt.show()

                        output_package["N_spectral_pixels"] = spec_wave[current_order_mask].size
                        output_package.ingest_data(optimization_output)

            if order_status != SUCCESS:
                # The order_status might chance inside the previous block!
                # remove the "obvious" measurements that come from here!
                output_package["RV"] = np.nan
                output_package["RV_uncertainty"] = np.nan
                output_package["N_spectral_pixels"] = np.nan
            output_package["status"] = order_status
            out_queue.put(output_package)
    except Exception as e:
        # guarantee that the shared buffers are closed from the worker side
        logger.opt(exception=True).critical("Worker is dead")
    finally:
        for buffer in shared_buffers:
            buffer.close()
        if natural_exit:
            out_queue.put(SHUTDOWN)
        else:
            out_queue.put(INTERNAL_ERROR)
