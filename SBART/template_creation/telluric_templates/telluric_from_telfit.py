import contextlib
import os
import tarfile
import urllib.request
from typing import Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
try:
    import telfit
    MISSING_TELFIT = False
except ModuleNotFoundError:
    MISSING_TELFIT = True

from scipy.optimize import minimize
from loguru import logger

from SBART import SBART_LOC
from SBART.ModelParameters import ModelComponent
from SBART.utils import custom_exceptions
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    Positive_Value_Constraint,
    StringValue,
    UserParam,
    ValueFromList,
)
from SBART.utils.paths_tools import file_older_than
from SBART.utils.status_codes import SUCCESS
from SBART.utils.types import UI_DICT
from SBART.utils.shift_spectra import (
    interpolate_data,
    remove_BERV_correction,
)
from SBART.utils.telluric_utilities.compute_order_overlap import (
    compute_wavelength_order_overlap,
)

from SBART.utils.units import kilometer_second
from .Telluric_Template import TelluricTemplate

atmospheric_profiles_coords_dict = {
    "HARPS": "-70.7-29.3",
    "ESPRESSO": "-70.4-24.6",
}


def download_gdas_archive(archive_name, storage_path):

    url = "https://ftp.eso.org/pub/dfs/pipelines/skytools/molecfit/gdas/"
    url += archive_name
    urllib.request.urlretrieve(url, storage_path)


@contextlib.contextmanager
def get_atmospheric_profile(instrument, datetime, storage_folder):
    try:
        coords = atmospheric_profiles_coords_dict[instrument]
    except KeyError as exc:
        raise custom_exceptions.InvalidConfiguration(f"Telfit template does not support {instrument}. Available instruments: {list(atmospheric_profiles_coords_dict.keys())}") from exc

    archive_name = f"gdas_profiles_C{coords}.tar.gz"
    profile_archive = os.path.join(storage_folder, f"gdas_profiles_C{coords}.tar.gz")
    # NOTE: these archives have thousands of files, maybe don't extract them
    exists = os.path.exists(profile_archive)
    if not exists or file_older_than(profile_archive, 7):  # !check if 7 is ok
        # download it from ESO
        logger.info("Downloading GDAS tar file from archive")
        download_gdas_archive(archive_name, profile_archive)

    msg = f"Looking for GDAS atmosphere profile in {profile_archive}"
    logger.info(msg)

    date, time = datetime.split("T")
    hour = int(time.split(":")[0])
    # GDAS profiles are available every 3 hours, find the closest one
    available = [0, 3, 6, 9, 12, 15, 18, 21]
    closest_hour = min(available, key=lambda list_value: abs(list_value - hour))

    gdas_file = f"C{coords}D{date}T{closest_hour:02d}.gdas"
    logger.info(f"Closest GDAS atmospheric profile is {gdas_file}")

    logger.info("Extracting file from the archive")
    tf = tarfile.open(profile_archive, "r:gz")

    try:
        yield tf.extractfile(gdas_file)
    finally:
        tf.close()


class TelfitTelluric(TelluricTemplate):
    """
    Create Earth's transmittance spectra from TelFit, configured with the night with the highest relative humidity.

    Atmosphere profiles are downloaded to ensure that we get the best model possible

    **User parameters:**

        - No unique user parameter


    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    .. warning::
        Telfit is not installed by default with SBART!

    """
    _default_params = TelluricTemplate._default_params + DefaultValues(
        atmosphere_profile=UserParam(
            "download", constraint=StringValue
        ),  # download / default / path
        FIT_MODEL=UserParam(False, constraint=BooleanValue),
        FIT_WAVELENGTH_STEP_SIZE=UserParam(0.001, constraint=Positive_Value_Constraint),
        # step size for telluric model wavelengths
    )
    _default_params.update("PARAMS_TO_FIT",
                           UserParam(
                               ["pressure", "humidity"],
                               constraint=ValueFromList(["temperature", "pressure", "humidity", "co2", "ch4", "n2o"]),
                           )
                           )
    method_name = "Telfit"

    def __init__(
        self,
        subInst: str,
        user_configs: Optional[UI_DICT] = None,
        extension_mode: str = "lines",
        application_mode: str = "removal",
        loaded: bool = False,
    ):
        if MISSING_TELFIT:
            raise custom_exceptions.InvalidConfiguration("Telfit is not currently installed")
        super().__init__(
            subInst=subInst,
            extension_mode=extension_mode,
            user_configs=user_configs,
            loaded=loaded,
            application_mode=application_mode,
        )

        model_components = []

        default_molecule_dict = dict(
            co2=385.34,
            ch4=1819.0,
            n2o=325.1,
        )

        model_components.append(
            ModelComponent(
                name="temperature", initial_guess=100, bounds=[0, None], default_enabled=True
            )
        )
        model_components.append(
            ModelComponent(
                name="pressure", initial_guess=100, bounds=[0, None], default_enabled=True
            )
        )
        model_components.append(
            ModelComponent(
                name="humidity", initial_guess=100, bounds=[0, None], default_enabled=True
            )
        )

        for name, value in default_molecule_dict.items():
            param = ModelComponent(
                name=name,
                initial_guess=value,
                bounds=[0, None],
                default_enabled=True, # FIXME: Why are they enabled?
                param_type="molecules::extra",
            )
            model_components.append(param)

        for comp in model_components:
            self._fitModel.add_extra_param(comp)

        self.modeler = telfit.Modeler(print_lblrtm_output=False)

    def _prepare_GDAS_data(self, selected_frame):

        logger.info("Preparing GDAS data load!")
        if selected_frame.is_Instrument("CARMENES") and (
            self._internal_configs["atmosphere_profile"] == "download"
            or self._internal_configs["force_download"]
        ):
            raise custom_exceptions.InvalidConfiguration(
                "We can't automate download from GDAS archive"
            )

        resources_folder = os.path.join(SBART_LOC, "resources/atmosphere_profiles")

        if self._internal_configs["atmosphere_profile"] == "download":
            # logger.debug("GDAS data load mode set to download")
            logger.info("Launching GDAS profile downloader")
            instrument, date = selected_frame.inst_name, selected_frame.get_KW_value("ISO-DATE")
            with get_atmospheric_profile(instrument, date, resources_folder) as gdas:
                data = np.loadtxt(gdas).copy()
            return data

        elif self._internal_configs["atmosphere_profile"] == "default":
            logger.info("Using the default atmosphere profile")
            atmos_profile_file = os.path.join(
                resources_folder,
                f"{selected_frame.inst_name}_atmosphere_profile.txt",
            )
            data = np.loadtxt(atmos_profile_file, usecols=(0, 1, 2, 3), skiprows=4)
            return data

        elif os.path.exists(self._internal_configs["atmosphere_profile"]) and not len(
            self._internal_configs["atmosphere_profile"] == 0
        ):
            atmos_profile_file = self._internal_configs["atmosphere_profile"]
        else:
            raise custom_exceptions.InvalidConfiguration()

        logger.info("Loading atmosphere profile from local file: {}", atmos_profile_file)
        data = np.loadtxt(atmos_profile_file)
        logger.info("Finished setup of GDAS profile")
        return data

    def configure_modeler(self, selected_frame) -> None:
        """
        see https://www.eso.org/sci/software/pipelines/skytools/molecfit#gdas
        """
        logger.info("Configuring the Telfit modeler for {}", selected_frame)
        data = self._prepare_GDAS_data(selected_frame)

        # hPa       m     K       %
        Pres, height, Temp, _ = data.T
        # to ºC
        Temp = Temp - 273.15
        # # calculate dew point
        # !not needed anymore...
        # a, b = 17.62, 243.12
        # α = lambda T, RH: (np.log(RH / 100) + a * T / (b + T))
        # dew = (b * α(Temp, relhum)) / (a - α(Temp, relhum))
        # dew[np.isnan(dew)] = -999

        # """Load the GDAS atmosphere profile following the recipe from
        # https://telfit.readthedocs.io/en/latest/GDAS_atmosphere.html
        # """
        # if self._configs["atmosphere_profile"] == "default":
        #     logger.info("Using default GDAS profile")
        #     atmosphere_filename = os.path.join(
        #         SBART_LOC,
        #         # f"resources/atmosphere_profiles/{self.instrument_info['name']}_atmosphere_profile.txt",
        #         f"resources/atmosphere_profiles/CARMENES_atmosphere_profile.txt",
        #     )
        # else:
        #     atmosphere_filename = self._configs["atmosphere_profile"]
        # if not os.path.exists(atmosphere_filename):
        #     raise FileNotFoundError(
        #         "GDAS profile does not exist in: {}".format(atmosphere_filename)
        #     )

        # logger.info("Loading GDAS atmosphere profile from {}".format(atmosphere_filename))

        # Pres, height, Temp, dew = np.loadtxt(
        #     atmosphere_filename, usecols=(0, 1, 2, 3), skiprows=4, unpack=True
        # )

        # Sort the arrays by height.
        sorter = np.argsort(height)
        height = height[sorter]
        Pres = Pres[sorter]
        Temp = Temp[sorter]
        # dew = dew[sorter]

        # Convert dew point temperature to ppmv
        Pw = 6.116441 * 10 ** (7.591386 * Temp / (Temp + 240.7263))
        h2o = Pw / (Pres - Pw) * 1e6

        # Unit conversion
        height = height / 1000.0
        Temp = Temp + 273.15
        self.modeler.EditProfile("Temperature", height, Temp)
        self.modeler.EditProfile("Pressure", height, Pres)
        self.modeler.EditProfile("H2O", height, h2o)

        logger.info("Finished configurating the modeler")

    def _generate_telluric_model(
        self, model_parameters, OBS_properties, fixed_params=None, grid=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """COmpute the transmittance spectra using tellfit

        Parameters
        ----------
        wavelengths : np.ndarray
            wavelengths for the reference observation

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Wavelengths and transmittance arrays
        """
        control_dict = {i: j for i, j in zip(self._fitModel.get_enabled_params(), model_parameters)}
        if fixed_params is not None:
            control_dict = {**control_dict, **fixed_params}

        if grid is None:
            # ask for a model slightly larger than the wavelength coverage of the instrument
            lowfreq = 1e7 / (OBS_properties["wavelength_coverage"][1] - 50)
            highfreq = 1e7 / (OBS_properties["wavelength_coverage"][0] + 50)
        else:
            lowfreq = 1e7 / grid.max()
            highfreq = 1e7 / grid.min()

        observatory = OBS_properties["EarthLocation"]
        m = self.modeler.MakeModel(
            lowfreq=lowfreq,
            highfreq=highfreq,
            vac2air=False,
            angle=np.rad2deg(np.arccos(1 / OBS_properties["airmass"])),
            lat=observatory.lat.value,
            alt=observatory.height.value / 1e3,
            resolution=OBS_properties["resolution"],
            wavegrid=grid,
            **control_dict,
        )
        return m.x, m.y

    @custom_exceptions.ensure_invalid_template
    def create_telluric_template(self, dataClass, custom_frameID: Optional[int] = None) -> None:
        """
        Create a telluric template from a TelFit transmission spectra [1], that
        was created for the date in which the reference observation was made.

        It estimates the continuum level and classifies each point that shows a
        decrease of 1% as a telluric line. Furthermore, it creates a window of
        6 points to each side of a detection, to attempt to pick up the wings of
        the telluric line.

        Parameters
        ----------
        dataClass:
            DataClass object
        custom_frameID :
            If Not None, does not search for the "optimal" frameID to use as a basis

        Returns
        -------
        numpy.ndarray
            Telluric (binary) spectrum, for the wavelengths present in the input
            array

        Notes
        -----------
        [1] https://github.com/kgullikson88/Telluric-Fitter
        """

        try:
            super().create_telluric_template(dataClass, custom_frameID=custom_frameID)
        except custom_exceptions.StopComputationError:
            return

        self.configure_modeler(dataClass.get_frame_by_ID(self._reference_frameID))

        OBS_properties = {
            **{"airmass": dataClass.get_KW_from_frameID("airmass", self._reference_frameID)},
            **dataClass.get_instrument_information(),
        }

        # Note: by default, when removing features, the "fit results" are the initial guesses
        #    Later on, if needed, this can be easily changed to fit before removing!

        parameter_values = self._fitModel.get_fit_results_from_frameID(
            frameID=self._reference_frameID, allow_disabled=True
        )
        names = self._fitModel.get_component_names(include_disabled=True)
        logger.info(f"Using params: {parameter_values}")

        wavelengths, tell_spectra = self._generate_telluric_model(
            model_parameters={},
            OBS_properties=OBS_properties,
            fixed_params=dict(zip(names, parameter_values)),
        )

        self.transmittance_wavelengths, self.transmittance_spectra = wavelengths, tell_spectra
        logger.info("Telfit model is complete.")

        # ! no median filtering (might still be needed in the future)
        self._continuum_level = 1.0
        self.wavelengths = wavelengths * 10  # convert to the prevalent wavelength units

        self._finish_template_creation()

    def _generate_model_parameters(self, dataClass):
        """
        Custom generation of the priors for the humidity, temperature and pressure
        Parameters
        ----------
        dataClass

        Returns
        -------

        """
        super()._generate_model_parameters(dataClass)

        for frameID in dataClass.get_frameIDs_from_subInst(
            self._associated_subInst, include_invalid=False
        ):
            frame = dataClass.get_frame_by_ID(frameID)
            initial_guess = {
                "humidity": frame.get_KW_value("relative_humidity"),
                "temperature": frame.get_KW_value("ambient_temperature"),
                "pressure": dataClass.get_instrument_information()["site_pressure"],
            }

            if not np.isfinite(initial_guess["humidity"]):
                initial_guess["humidity"] = 25
                logger.warning(
                    "Relative humidity is not finite. Using default value of {}%", initial_guess["humidity"]
                )

            if not np.isfinite(initial_guess["temperature"]):
                initial_guess["temperature"] = 290.5
                logger.warning(
                    "Ambient temperature is not finite. Using default value of {}K", initial_guess["temperature"]
                )

            self._fitModel.update_params_initial_guesses(
                frameID=frame.frameID, guesses=initial_guess
            )

            if self.for_feature_removal:
                finals = [
                    comps.get_initial_guess(frameID)
                    for comps in self._fitModel.get_enabled_components()
                ]
                # As we won't be fitting the parameters, we can already update its "final value"
                self._fitModel.store_frameID_results(frameID, finals, result_flag=SUCCESS)

    def fit_telluric_model_to_frame(self, frame):
        super().fit_telluric_model_to_frame(frame)
        if not frame.is_blaze_corrected:
            ...
            # raise custom_exceptions.InvalidConfiguration("When correcting tellurics must have the frames with a BLAZE correction")

        for param in self._fitModel.get_enabled_components():
            if param.is_parameter_type("molecules::extra"):
                raise custom_exceptions.InvalidConfiguration(
                    "We can only fit H20. Please don't enable extra molecules"
                )

        wavelengths, spectra, uncertainties, mask = frame.get_data_from_full_spectrum()
        wavelengths /= 10

        self.configure_modeler(frame)
        rest_frame_wavelengths = remove_BERV_correction(
            wavelengths, frame.get_KW_value("BERV").to(kilometer_second).value
        )

        OBS_properties = {
            **{"airmass": frame.get_KW_value("airmass")},
            **frame.instrument_properties,
        }

        # Select wavelength region (to account for the 20 water lines)
        lines_to_fit = [
            [7178.38, 7180.30],
            [7182.55, 7184.47],
            [7188.42, 7190.34],
            [7192.51, 7194.43],
            [7194.78, 7196.70],
            [7202.22, 7204.14],
            [7205.32, 7207.24],
            [7224.65, 7226.58],
            [7241.66, 7243.59],
            [7244.74, 7246.67],
            [7246.70, 7248.64],
            [7253.41, 7255.34],
            [7266.62, 7268.56],
            [7273.99, 7275.93],
            [7276.44, 7278.38],
            [7278.42, 7280.36],
            [7288.42, 7290.36],
            [7291.42, 7293.37],
            [7305.23, 7307.18],
            [7310.55, 7312.50],
        ]

        lines_to_fit = np.divide(lines_to_fit, 10).tolist()
        flat_list = [item for sublist in lines_to_fit for item in sublist]
        region_to_fit = (min(flat_list) - 0.5, max(flat_list) + 0.5)

        step = self._internal_configs["FIT_WAVELENGTH_STEP_SIZE"]
        model_gen_grid = np.arange(region_to_fit[0], region_to_fit[1] + step, step)
        logger.info("Using grid with {} points", model_gen_grid.size)
        overlapping_orders = compute_wavelength_order_overlap(wavelengths, region_to_fit)

        line_information = []
        logger.info(
            "Fitting telluric features across {} slices: {}".format(
                len(overlapping_orders), overlapping_orders
            )
        )

        for order in overlapping_orders:
            wavelengths_order = rest_frame_wavelengths[order]
            spectra_order = spectra[order]

            order_lines = []
            order_wavelengths = []
            order_blaze = []
            for line in lines_to_fit:
                inds = np.where(
                    np.logical_and(wavelengths_order >= line[0], wavelengths_order <= line[1])
                )
                if len(inds[0]) == 0:
                    continue
                else:
                    print(len(inds[0]))
                    # order_blaze.append(BLAZE[order][inds])
                    order_lines.append(spectra_order[inds])
                    order_wavelengths.append(wavelengths_order[inds])

            from scipy.ndimage import median_filter
            inds = slice(0, 10000)
            for i in range(1):

                n_points_filter = 500
                continuum_level = median_filter(spectra[order][inds], n_points_filter)
                continuum_level[0: n_points_filter + 1] = median_filter(
                    spectra[order][inds][0: n_points_filter + 1], 51
                )
                continuum_level[-(n_points_filter + 1):] = median_filter(
                    spectra[order][inds][-(n_points_filter + 1):], 51
                )

                inds = np.where(spectra[order][inds] != 10*continuum_level)

            params = np.polyfit(wavelengths_order[inds], spectra[order][inds], 4)
            p = np.poly1d(params)
            # line_information.extend(list(zip(order_wavelengths, order_lines, order_blaze)))
            for waves, spec in zip(order_wavelengths, order_lines):
                print(waves, spec)
                plt.scatter(waves, spec, color = "orange", zorder=1000, marker='x')

            plt.title(order)
            plt.plot(wavelengths_order, spectra[order], color="black")
            plt.plot(wavelengths_order, p(wavelengths_order), color = 'red', linestyle='--')
            plt.plot(wavelengths_order, continuum_level, color = 'blue', linestyle='--')

            plt.show()
        final_vector, bounds = self._fitModel.generate_optimizer_inputs(
            frameID=frame.frameID, rv_units=None  # everything is floats
        )

        logger.info(
            "{} - {} - {}".format(self._fitModel.get_enabled_params(), final_vector, bounds)
        )

        if self.was_loaded:
            # TODO: implement the save/load routines
            final_vector = self._fitModel.get_fit_results_from_frameID(frame.frameID)
        else:

            initial_guess, bounds = self._fitModel.generate_optimizer_inputs(
                frameID=frame.frameID, rv_units=None  # everything is floats
            )

            fit_fixed_params = {}
            mandatory_params = ["temperature", "pressure", "humidity"]

            free_params = self._fitModel.get_enabled_params()
            logger.debug(f"Fitting telfit model with <{free_params}> as free-parameters")

            for entry in mandatory_params:
                if entry not in free_params:
                    if entry != "temperature":
                        # We don't really care about having the temperature as a fixed parameter, as
                        # it has a very small impact on the depth/shape of telluric features

                        logger.warning("Fitting telfit model with {} as a fixed parameter", entry)
                    fit_fixed_params[entry] = self._fitModel.get_initial_guess_of_component(
                                 entry, frameID=frame.frameID, allow_disabled=True)

            if len(fit_fixed_params) == 0:
                fit_fixed_params = None

            min_results = minimize(
                self._fit_tell_model,
                x0=initial_guess,
                args=(model_gen_grid, OBS_properties, line_information, fit_fixed_params),
            )
            if not min_results.success:
                logger.warning("{} fit has failed!", self.name)

            final_vector = min_results.x
            # TODO: ensure that the BERVs corrections are matching up

            self._fitModel.store_frameID_results(frameID=frame.frameID, result_vector=final_vector)

        # Finished fitting. Generating final model, outside the selected lines

        fixed_params = {
            comp.param_name: comp.get_initial_guess(frame.frameID, allow_disabled=True)
            for comp in self._fitModel.get_disabled_components()
        }

        # TODO: create model for all orders!
        wavelengths, model = self._generate_telluric_model(
            grid=None,
            model_parameters=final_vector,
            OBS_properties=OBS_properties,
            fixed_params=fixed_params,
        )
        model_uncertainties = model  # for now we assume the model to be noise free -> direct re-scaling of uncertainties
        plt.figure()

        plt.plot(wavelengths, model)
        plt.show()
        return wavelengths, model, model_uncertainties

    def _fit_tell_model(
        self, tentative_param_vector, new_grid, OBS_properties, enabled_lines, fixed_params
    ):
        wavelengths, model = self._generate_telluric_model(
            grid=new_grid,
            model_parameters=tentative_param_vector,
            OBS_properties=OBS_properties,
            fixed_params=fixed_params,
        )

        return cost_function(wavelengths, model, line_info=enabled_lines)


def cost_function(model_wavelengths, model_transmittance, line_info):
    """
    Cost function for the fit!

    Parameters
    ----------
    model_wavelengths
        Telfit model wavelengths
    model_transmittance
        Telfit model transmittance
    line_info
        Lines that are being tracked

    Returns
    -------

    """
    metric = []
    fig, axis = plt.subplots(3, 1, sharex=True)
    # plt.plot(model_wavelengths, model_transmittance)
    axis[0].plot(model_wavelengths, model_transmittance, color="blue", alpha=0.3, linestyle="--")
    axis[1].plot(model_wavelengths, model_transmittance, color="blue", alpha=0.3, linestyle="--")
    raw = []
    BLAZE_CORR = []
    for line in line_info:
        waves = line[0]
        spec = line[1]
        interpol_model, errors, indexes = interpolate_data(
            original_lambda=model_wavelengths,
            original_spectrum=model_transmittance,
            new_lambda=waves,
            lower_limit=-np.inf,
            upper_limit=np.inf,
            original_errors=[],
            propagate_interpol_errors="none",
        )
        raw.extend(spec)
        metric.extend(spec / interpol_model)

        axis[0].plot(waves, spec, color="black")
        axis[1].plot(waves, interpol_model, color="blue")

        axis[0].plot(waves, spec / interpol_model, color="red", linestyle="--")
        blaze_corr_res = (spec / interpol_model) / line[2]
        axis[2].plot(waves, blaze_corr_res)

        BLAZE_CORR.extend(blaze_corr_res - np.median(blaze_corr_res))
    print(np.std(metric), np.std(raw), np.std(BLAZE_CORR))
    plt.show()

    return np.std(metric)
