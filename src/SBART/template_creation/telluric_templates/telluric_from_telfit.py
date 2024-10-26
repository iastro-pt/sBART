import contextlib
import os
import tarfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

try:
    import telfit

    MISSING_TELFIT = False
except ModuleNotFoundError:
    MISSING_TELFIT = True

from loguru import logger

from SBART import SBART_LOC
from SBART.internals.cache import DB_connection
from SBART.ModelParameters import ModelComponent
from SBART.utils import custom_exceptions
from SBART.utils.paths_tools import file_older_than
from SBART.utils.status_codes import SUCCESS
from SBART.utils.types import UI_DICT
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    NumericValue,
    Positive_Value_Constraint,
    StringValue,
    UserParam,
    ValueFromList,
)

from .Telluric_Template import TelluricTemplate

atmospheric_profiles_coords_dict = {
    "HARPS": "-70.7-29.3",
    "NIRPS": "-70.7-29.3",
    "ESPRESSO": "-70.4-24.6",
    "HARPSN": "-17.9+28.8",
    "CARMENES": "-2.5+37.2",
}


def construct_gdas_filename(instrument, datetime):
    """Construct the filename as in GDAS archive

    Args:
        instrument (str): Instrument name
        datetime (str): Date and time, as in the header of fits files

    Raises:
        custom_exceptions.InvalidConfiguration: _description_

    Returns:
        _type_: _description_

    """
    try:
        coords = atmospheric_profiles_coords_dict[instrument]
    except KeyError as exc:
        raise custom_exceptions.InvalidConfiguration(
            f"Telfit template does not support {instrument}. Available instruments: {list(atmospheric_profiles_coords_dict.keys())}",
        ) from exc

    date, time = datetime.split("T")
    hour = int(time.split(":")[0])
    # GDAS profiles are available every 3 hours, find the closest one
    available = [0, 3, 6, 9, 12, 15, 18, 21]
    closest_hour = min(available, key=lambda list_value: abs(list_value - hour))

    return f"C{coords}D{date}T{closest_hour:02d}.gdas"


def download_gdas_archive(archive_name, storage_path):
    url = "https://ftp.eso.org/pub/dfs/pipelines/skytools/molecfit/gdas/"
    url += archive_name
    urllib.request.urlretrieve(url, storage_path)


@contextlib.contextmanager
def get_atmospheric_profile(instrument, datetime, storage_folder):
    try:
        coords = atmospheric_profiles_coords_dict[instrument]
    except KeyError as exc:
        raise custom_exceptions.InvalidConfiguration(
            f"Telfit template does not support {instrument}. Available instruments: {list(atmospheric_profiles_coords_dict.keys())}",
        ) from exc

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

    gdas_file = construct_gdas_filename(instrument=instrument, datetime=datetime)

    logger.info(f"Closest GDAS atmospheric profile is {gdas_file}")

    logger.info("Extracting file from the archive")
    tf = tarfile.open(profile_archive, "r:gz")

    try:
        yield tf.extractfile(gdas_file)
    finally:
        tf.close()


class TelfitTelluric(TelluricTemplate):
    """Create Earth's transmittance spectra from TelFit, configured with the night with the highest
    relative humidity.

    Atmosphere profiles are downloaded to ensure that we get the best model possible

    **User parameters:**
    ================================ ================ ================ ================ ================
    Parameter name                      Mandatory      Default Value    Valid Values    Comment
    ================================ ================ ================ ================ ================
    TELFIT_HUMIDITY_THRESHOLD            False          None            Numerical        [1]
    ================================ ================ ================ ================ ================


    .. note::
        [1] - Value used to enforce a maximum value for the humidity. If this is set any negative value, use the
        maximum humidity from the loaded observations


    .. note::
        Also check the **User parameters** of the parent classes for further customization options of SBART

    .. warning::
        Telfit is not installed by default with SBART!

    """

    _default_params = TelluricTemplate._default_params + DefaultValues(
        atmosphere_profile=UserParam("GDAS", constraint=StringValue),  # GDAS / default / path
        FIT_MODEL=UserParam(False, constraint=BooleanValue),
        TELFIT_HUMIDITY_THRESHOLD=UserParam(default_value=-1, constraint=NumericValue),
        FIT_WAVELENGTH_STEP_SIZE=UserParam(0.001, constraint=Positive_Value_Constraint),
        # step size for telluric model wavelengths
        PARAMS_TO_FIT=UserParam(
            ["pressure", "humidity"],
            constraint=ValueFromList(["temperature", "pressure", "humidity", "co2", "ch4", "n2o"]),
        ),
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
                name="temperature",
                initial_guess=100,
                bounds=[0, None],
                default_enabled=True,
            ),
        )
        model_components.append(
            ModelComponent(
                name="pressure",
                initial_guess=100,
                bounds=[0, None],
                default_enabled=True,
            ),
        )
        model_components.append(
            ModelComponent(
                name="humidity",
                initial_guess=100,
                bounds=[0, None],
                default_enabled=True,
            ),
        )

        for name, value in default_molecule_dict.items():
            param = ModelComponent(
                name=name,
                initial_guess=value,
                bounds=[0, None],
                default_enabled=True,
                param_type="molecules::extra",
            )
            model_components.append(param)

        for comp in model_components:
            self._fitModel.add_extra_param(comp)

        self.modeler = telfit.Modeler(print_lblrtm_output=False)

    def _prepare_GDAS_data(self, dataClass, selected_frame):  # pylint: disable=C0103
        logger.info("Preparing GDAS data load!")

        resources_folder = os.path.join(SBART_LOC, "resources/atmosphere_profiles")

        conn = DB_connection()

        if self._internal_configs["atmosphere_profile"] == "GDAS":
            # logger.debug("GDAS data load mode set to download")
            logger.info("Launching GDAS profile downloader")
            instrument, date = selected_frame.inst_name, selected_frame.get_KW_value("ISO-DATE")

            logger.warning("Iterating over other possible frames to search for a working reference")
            failed_tests = [self._reference_frameID]
            for kw in ["relative_humidity", "airmass"]:
                metric_to_select, frameIDs = dataClass.collect_KW_observations(
                    KW=kw,
                    subInstruments=[self._associated_subInst],
                    include_invalid=False,
                    conditions=self._metric_selection_conditions,
                    return_frameIDs=True,
                )
                metric_to_select = np.asarray(metric_to_select, dtype=float)
                metric_to_select = list(i if i is not None else np.nan for i in metric_to_select)
                if not any(np.isfinite(metric_to_select)):
                    logger.warning(f"Metric {kw} is not finite. Can't use it to select observatioons")
                    continue
                # Stop if we have finite values!
                break
            else:
                logger.info(f"Using {kw} as the metric to select transmittance profile")

            sort_inds = np.argsort(metric_to_select)
            frames_to_search = np.asarray(frameIDs)[sort_inds][::-1]

            max_search_iterations = min(10, len(frameIDs))
            logger.info("Starting loop to retrive GDAS profile")

            for attempt_nb in range(max_search_iterations):  # TODO: add here the search for a new reference!
                selected_ID = frames_to_search[0]

                date = dataClass.get_KW_from_frameID(frameID=selected_ID, KW="ISO-DATE")

                gdas_filename = construct_gdas_filename(instrument=instrument, datetime=date)
                try:
                    data = conn.get_GDAS_profile(gdas_filename=gdas_filename)
                    logger.info("Using cached version of the GDAS profile")
                    break
                except FileNotFoundError:
                    pass

                try:
                    with get_atmospheric_profile(instrument, date, resources_folder) as gdas:
                        data = np.loadtxt(gdas).copy()
                    self._reference_frameID = selected_ID
                    self._associated_BERV = dataClass.get_KW_from_frameID(KW="BERV", frameID=selected_ID)
                    conn.add_new_profile(gdas_filename=gdas_filename, data=data, instrument=instrument)
                    break  # If we found it, no need to continue
                except KeyError:
                    logger.info(
                        f"{date=} failed to find GDAS profile (Iter: {attempt_nb + 1} / {max_search_iterations})",
                    )
                    failed_tests.append(selected_ID)
                    frames_to_search = frames_to_search[1:]
            else:
                logger.warning("Couldn't download any of the GDAS profiles. Moving on for the default profile")

        if self._internal_configs["atmosphere_profile"] == "default":
            logger.warning("Using the default atmosphere profile!")
            atmos_profile_file = os.path.join(
                resources_folder,
                f"{selected_frame.inst_name}_atmosphere_profile.txt",
            )
            data = np.loadtxt(atmos_profile_file)

        elif os.path.exists(self._internal_configs["atmosphere_profile"]) and not len(
            self._internal_configs["atmosphere_profile"] == 0,
        ):
            # what does this do???
            atmos_profile_file = self._internal_configs["atmosphere_profile"]
            data = np.loadtxt(atmos_profile_file)

        logger.info("Finished setup of GDAS profile")
        return data

    def configure_modeler(self, dataclass) -> None:
        """See https://www.eso.org/sci/software/pipelines/skytools/molecfit#gdas
        """
        selected_frame = dataclass.get_frame_by_ID(self._reference_frameID)
        logger.info("Configuring the Telfit modeler for {}", selected_frame)
        data = self._prepare_GDAS_data(dataClass=dataclass, selected_frame=selected_frame)

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
        self, model_parameters, OBS_properties, fixed_params=None, grid=None,
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
            lowfreq = 1e7 / (OBS_properties["wavelength_coverage"][1] + 50)
            highfreq = 1e7 / (OBS_properties["wavelength_coverage"][0] - 50)
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
        """Create a telluric template from a TelFit transmission spectra [1], that
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
        -----
        [1] https://github.com/kgullikson88/Telluric-Fitter

        """
        try:
            super().create_telluric_template(dataClass, custom_frameID=custom_frameID)
        except custom_exceptions.StopComputationError:
            return

        self.configure_modeler(dataClass)

        OBS_properties = {
            "airmass": dataClass.get_KW_from_frameID("airmass", self._reference_frameID),
            **dataClass.get_instrument_information(),
        }

        # Note: by default, when removing features, the "fit results" are the initial guesses
        #    Later on, if needed, this can be easily changed to fit before removing!

        parameter_values = self._fitModel.get_fit_results_from_frameID(
            frameID=self._reference_frameID, allow_disabled=True,
        )
        names = self._fitModel.get_component_names(include_disabled=True)
        logger.debug(f"Parameters in use: {names}")
        logger.info(f"Using params: {parameter_values}")
        wavelengths, tell_spectra = self._generate_telluric_model(
            model_parameters={},
            OBS_properties=OBS_properties,
            fixed_params=dict(zip(names, parameter_values)),
        )

        self.transmittance_wavelengths, self.transmittance_spectra = (
            wavelengths,
            tell_spectra,
        )
        logger.info("Telfit model is complete.")

        # ! no median filtering (might still be needed in the future)
        self._continuum_level = 1.0
        self.wavelengths = wavelengths * 10  # convert to the prevalent wavelength units

        self._finish_template_creation()

    def _generate_model_parameters(self, dataClass):
        """Custom generation of the priors for the humidity, temperature and pressure

        Parameters
        ----------
        dataClass

        Returns
        -------

        """
        super()._generate_model_parameters(dataClass)

        for frameID in dataClass.get_frameIDs_from_subInst(self._associated_subInst, include_invalid=False):
            frame = dataClass.get_frame_by_ID(frameID)
            initial_guess = {
                "humidity": frame.get_KW_value("relative_humidity"),
                "temperature": frame.get_KW_value("ambient_temperature"),
                "pressure": dataClass.get_instrument_information()["site_pressure"],
            }

            if not np.isfinite(initial_guess["humidity"]):
                initial_guess["humidity"] = 25
                logger.warning(
                    "Relative humidity is not finite. Using default value of {}%",
                    initial_guess["humidity"],
                )

            if 0 <= self._internal_configs["TELFIT_HUMIDITY_THRESHOLD"] < initial_guess["humidity"]:
                initial_guess["humidity"] = self._internal_configs["TELFIT_HUMIDITY_THRESHOLD"]
                user_cap = self._internal_configs["TELFIT_HUMIDITY_THRESHOLD"]
                curr_val = initial_guess["humidity"]
                logger.warning(
                    f"Relative humidity ({curr_val}) is above the user-provided threshold ({user_cap}). Falling back to it",
                )

            if not np.isfinite(initial_guess["temperature"]):
                initial_guess["temperature"] = 290.5
                logger.warning(
                    "Ambient temperature is not finite. Using default value of {}K",
                    initial_guess["temperature"],
                )

            self._fitModel.update_params_initial_guesses(frameID=frame.frameID, guesses=initial_guess)

            if self.for_feature_removal:
                finals = [comps.get_initial_guess(frameID, True) for comps in self._fitModel.get_enabled_components()]
                self._fitModel.store_frameID_results(frameID, finals, result_flag=SUCCESS)

        for param_name in self._fitModel.get_enabled_params():
            if param_name not in self._internal_configs["PARAMS_TO_FIT"]:
                logger.info(
                    "{} not fitting {}. Fixing it to initial guess",
                    self.name,
                    param_name,
                )
                self._fitModel.disable_param(param_name)

    def store_metrics(self):
        super().store_metrics()
        metrics_path = self._internalPaths.get_path_to("metrics", as_posix=False)
        parameter_values = self._fitModel.get_fit_results_from_frameID(
            frameID=self._reference_frameID, allow_disabled=True,
        )
        names = self._fitModel.get_component_names(include_disabled=True)

        with open(metrics_path / f"telfit_info_{self._associated_subInst}.txt", mode="w") as to_write:
            for nam, val in zip(names, parameter_values):
                to_write.write(f"{nam}:  {val}\n")
