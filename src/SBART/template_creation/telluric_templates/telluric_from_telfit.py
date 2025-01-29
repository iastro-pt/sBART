from __future__ import annotations

import contextlib
import os
import tarfile
import urllib.request
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger

from SBART import SBART_LOC
from SBART.internals.cache import DB_connection
from SBART.ModelParameters import ModelComponent
from SBART.utils import custom_exceptions
from SBART.utils.choices import DISK_SAVE_MODE
from SBART.utils.paths_tools import file_older_than
from SBART.utils.spectral_conditions import KEYWORD_condition
from SBART.utils.status_codes import SUCCESS
from SBART.utils.telluric_utilities import create_binary_template
from SBART.utils.UserConfigs import (
    BooleanValue,
    DefaultValues,
    NumericValue,
    Positive_Value_Constraint,
    StringValue,
    UserParam,
    ValueFromDtype,
    ValueFromList,
)

from .Telluric_Template import TelluricTemplate

if TYPE_CHECKING:
    from SBART.data_objects.DataClass import DataClass
    from SBART.utils.SBARTtypes import UI_DICT

RESOURCES_PATH = Path(__file__).parent.parent.parent / "resources"

atmospheric_profiles_coords_dict = {
    "HARPS": "-70.7-29.3",
    "NIRPS": "-70.7-29.3",
    "ESPRESSO": "-70.4-24.6",
    "HARPSN": "-17.9+28.8",
    "CARMENES": "-2.5+37.2",
    "CALIB_CARMENES": "-2.5+37.2",
}


def fun(params, modeler, loc, OBS_properties, freqs, control_dict, grid) -> None:
    airmass, humidity, temperature = params

    fname = f"transmittance_{airmass:.1f}_{humidity:.0f}_{temperature:.0f}.txt"

    lowfreq, highfreq = freqs
    observatory = OBS_properties["EarthLocation"]

    control_dict["temperature"] = temperature
    control_dict["humidity"] = humidity

    m = modeler.MakeModel(
        lowfreq=lowfreq,
        highfreq=highfreq,
        vac2air=False,
        angle=np.rad2deg(np.arccos(1 / airmass)),
        lat=observatory.lat.value,
        alt=observatory.height.value / 1e3,
        resolution=OBS_properties["resolution"],
        wavegrid=grid,
        **control_dict,
    )
    wave, transmit = m.x, m.y
    np.savetxt(fname=loc / fname, X=np.c_[wave, transmit])


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

    _default_params = (
        TelluricTemplate._default_params
        + DefaultValues(
            atmosphere_profile=UserParam("GDAS", constraint=StringValue),  # GDAS / default / path
            FIT_MODEL=UserParam(False, constraint=BooleanValue),
            TELFIT_HUMIDITY_THRESHOLD=UserParam(default_value=-1, constraint=NumericValue),
            FIT_WAVELENGTH_STEP_SIZE=UserParam(0.001, constraint=Positive_Value_Constraint),
            # step size for telluric model wavelengths
            PARAMS_TO_FIT=UserParam(
                ["pressure", "humidity"],
                constraint=ValueFromList(["temperature", "pressure", "humidity", "co2", "ch4", "n2o"]),
            ),
            USE_GRID_OF_TRANSMITTANCE=UserParam(
                default_value=False,
                constraint=BooleanValue,
                description=(
                    "If True (default False), uses a grid of pre-computed Telfit transmittances to generate"
                    "the telluric model. If the grid doesn't exist, it will generate a new one"
                ),
            ),
            GRID_MAIN_PATH=UserParam(
                default_value=None,
                constraint=ValueFromDtype((str, Path, type(None))),
                description=(
                    "If not None, it will be used to store the transmittance grid. If None"
                    "stores in s_BART 'resources' folder in the location of installation"
                ),
            ),
            IND_WATER_MASK_THRESHOLD=UserParam(  # Ensuring that things don't blow up when storing the fits files (inf will do that)
                default_value=1e8,
                constraint=Positive_Value_Constraint,
                description="Independent masking of water features, using a different threshold than for other molecules",
            ),
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
        super().__init__(
            subInst=subInst,
            extension_mode=extension_mode,
            user_configs=user_configs,
            loaded=loaded,
            application_mode=application_mode,
        )

        model_components = []

        default_molecule_dict = {
            "co2": 385.34,
            "ch4": 1819.0,
            "n2o": 325.1,
        }

        model_components.append(
            ModelComponent(
                name="temperature",
                initial_guess=283.15,
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

        self.modeler = None

    def _prepare_GDAS_data(self, dataClass, selected_frame):  # pylint: disable=C0103
        logger.info("Preparing GDAS data load!")

        resources_folder = SBART_LOC / "resources/atmosphere_profiles"

        conn = DB_connection()

        if self._internal_configs["atmosphere_profile"] == "GDAS":
            logger.info("Launching GDAS profile downloader")
            instrument, date = selected_frame.inst_name, selected_frame.get_KW_value("ISO-DATE")

            # JD for '2004-12-01T00:00:00', which is the same as the first profile from GDAS
            self._metric_selection_conditions += KEYWORD_condition("BJD", [2453340.5, np.inf])

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

            max_search_iterations = len(frameIDs)
            logger.info("Starting loop to retrive GDAS profile")

            found = True
            for attempt_nb in range(max_search_iterations):
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
                found = False

        if self._internal_configs["atmosphere_profile"] == "default" or not found:
            logger.warning("Using the default atmosphere profile!")
            atmos_profile_file = resources_folder / f"{selected_frame.inst_name}_atmosphere_profile.txt"
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
        """See https://www.eso.org/sci/software/pipelines/skytools/molecfit#gdas"""
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
        self,
        dataClass,
        model_parameters,
        OBS_properties,
        instrument: str,
        fixed_params=None,
        grid=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """COmpute the transmittance spectra using tellfit.

        Parameters
        ----------
        wavelengths : np.ndarray
            wavelengths for the reference observation

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Wavelengths and transmittance arrays

        """
        control_dict = dict(zip(self._fitModel.get_enabled_params(), model_parameters))
        if fixed_params is not None:
            control_dict = {**control_dict, **fixed_params}

        if grid is None:
            # ask for a model slightly larger than the wavelength coverage of the instrument
            lowfreq = 1e7 / (OBS_properties["wavelength_coverage"][1] + 50)
            highfreq = 1e7 / (OBS_properties["wavelength_coverage"][0] - 50)
        else:
            lowfreq = 1e7 / grid.max()
            highfreq = 1e7 / grid.min()

        generate_new_grid = False
        if self._internal_configs["USE_GRID_OF_TRANSMITTANCE"]:
            if self._internal_configs["GRID_MAIN_PATH"] is not None:
                loc = Path(self._internal_configs["GRID_MAIN_PATH"])
            else:
                loc = RESOURCES_PATH / "atmosphere_grid" / instrument
                if not loc.exists() or len(list(loc.iterdir())) == 0:
                    logger.warning("Folder with grids is not available. Generating it now")
                    loc.mkdir(exist_ok=True, parents=True)
                    # Not the greatest code, but this ensures that the modeler
                    # is ready to go when we launch the telfit grid computation

                    try:
                        import telfit

                    except ModuleNotFoundError as e:
                        msg = "Telfit is not istalled"
                        raise custom_exceptions.InternalError(msg) from e

                    if self.modeler is None:
                        self.modeler = telfit.Modeler(print_lblrtm_output=False)
                        self.configure_modeler(dataClass)

                    self._construct_grid(
                        loc,
                        OBS_properties=OBS_properties,
                        control_dict=control_dict,
                        freqs=[lowfreq, highfreq],
                        grid=None,  # TODO: see if we can reduce the number of points in here
                    )
            logger.debug(f"Using grid of trasmittance values, searching in {loc}")
            # This needs to be updated if the grid changes
            closest_temp = int(np.ceil(control_dict["temperature"] / 2) * 2)
            closest_rhum = int(np.ceil(control_dict["humidity"] / 5) * 5)
            closest_airmass = np.round(np.ceil(OBS_properties["airmass"] / 0.1) * 0.1, 2)

            desired_file = loc / f"transmittance_{closest_airmass}_{closest_rhum}_{closest_temp}.txt"
            if not desired_file.exists():
                params = (closest_temp, closest_rhum, closest_airmass)

                logger.critical(
                    f"Using a grid of transmittance and missing the relevant profiles ({params}). Moving onwards to nominal run"
                )
            else:
                logger.debug("Found pre-computed transmittance spectra, serving it")
                data = np.loadtxt(desired_file)
                return data[:, 0], data[:, 1]

        try:
            import telfit

        except ModuleNotFoundError as e:
            msg = "Telfit is not istalled"
            raise custom_exceptions.InternalError(msg) from e

        if self.modeler is None:
            self.modeler = telfit.Modeler(print_lblrtm_output=False)
            self.configure_modeler(dataClass)

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

    def _construct_grid(self, loc, OBS_properties, control_dict, freqs, grid):
        # If we update the grid steps, we will need to also change the search for the closest file
        airmass_values = np.arange(start=1, stop=2.1, step=0.1)
        humidity_values = np.arange(start=0, stop=35, step=5, dtype=int)
        temperature_values = np.arange(278, 294.1, step=2, dtype=int)  # from 5 to 20 Celsius

        combinations = product(airmass_values, humidity_values, temperature_values)

        target = partial(
            fun,
            modeler=self.modeler,
            loc=loc,
            OBS_properties=OBS_properties,
            freqs=freqs,
            control_dict=control_dict,
            grid=grid,
        )

        with Pool(processes=4) as p:
            p.map(target, combinations)

    @custom_exceptions.ensure_invalid_template
    def create_telluric_template(self, dataClass: DataClass, custom_frameID: Optional[int] = None) -> None:
        """Create a telluric template from a TelFit transmission spectra [1].

        The model is created for the date in which the reference observation was made.

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

        OBS_properties = {
            "airmass": dataClass.get_KW_from_frameID("airmass", self._reference_frameID),
            **dataClass.get_instrument_information(),
        }

        # Note: by default, when removing features, the "fit results" are the initial guesses
        #    Later on, if needed, this can be easily changed to fit before removing!

        parameter_values = self._fitModel.get_fit_results_from_frameID(
            frameID=self._reference_frameID,
            allow_disabled=True,
        )
        instrument = dataClass.get_frame_by_ID(self._reference_frameID).inst_name

        names = self._fitModel.get_component_names(include_disabled=True)
        logger.debug(f"Parameters in use: {names}")
        logger.info(f"Using params: {parameter_values}")

        # This is the model with every molecule in the dataset. If there is no individual masking for water
        # This will work with the same threshold for every single element
        threshold = self._internal_configs["continuum_percentage_drop"]

        USE_INDEPENDENT_WATER = self._internal_configs["IND_WATER_MASK_THRESHOLD"] < 1e6
        if USE_INDEPENDENT_WATER:
            logger.debug("There is water")

            threshold = self._internal_configs["IND_WATER_MASK_THRESHOLD"]

        wavelengths, tell_spectra = self._generate_telluric_model(
            dataClass=dataClass,
            model_parameters={},
            OBS_properties=OBS_properties,
            fixed_params=dict(zip(names, parameter_values)),
            instrument=instrument,
        )

        self.template = create_binary_template(
            transmittance=tell_spectra,
            continuum_level=1.0,
            percentage_drop=threshold,
        )

        # ! no median filtering (might still be needed in the future)
        self._continuum_level = 1.0
        self.wavelengths = wavelengths * 10  # convert to the prevalent wavelength units

        self.transmittance_wavelengths, self.transmittance_spectra = (
            wavelengths,
            tell_spectra,
        )

        if USE_INDEPENDENT_WATER:
            # We wanted an individual mask for water, which means that now we have to
            # create the mask with the correct transmittance for the other elements
            logger.debug("Starting full spectra model")
            parameters = {}
            for key, value in zip(names, parameter_values):
                parameters[key] = value
                if key == "humidity":
                    parameters[key] = 0

            logger.warning(f"Original value: {tell_spectra.shape}")

            waves, tell_spectra = self._generate_telluric_model(
                dataClass=dataClass,
                model_parameters={},
                OBS_properties=OBS_properties,
                fixed_params=parameters,
                grid=wavelengths,
                instrument=instrument,
            )

            logger.warning(f"New value: {tell_spectra.shape}")
            self.template += create_binary_template(
                transmittance=tell_spectra,
                continuum_level=1.0,
                percentage_drop=self._internal_configs["continuum_percentage_drop"],
            )

            # Ensure that we don't have values grater than 1, to keep consistency
            self.template[np.where(self.template > 1)] = 1

        self.build_blocks()
        self._compute_wave_blocks()
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

        if self.for_feature_correction or self.disk_save_level == DISK_SAVE_MODE.DISABLED:
            metrics_path = self._internalPaths.get_path_to("metrics", as_posix=False)
            parameter_values = self._fitModel.get_fit_results_from_frameID(
                frameID=self._reference_frameID,
                allow_disabled=True,
            )
            names = self._fitModel.get_component_names(include_disabled=True)

            with open(metrics_path / f"telfit_info_{self._associated_subInst}.txt", mode="w") as to_write:
                for nam, val in zip(names, parameter_values):
                    to_write.write(f"{nam}:  {val}\n")
