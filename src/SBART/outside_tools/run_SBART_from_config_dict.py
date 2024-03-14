import math
import os

from SBART.Instruments import instrument_dict as instrument_name_map
from SBART.Quality_Control.activity_indicators import Indicators
from SBART.Samplers import Sampler_map
from SBART.data_objects import DataClassManager
from SBART.data_objects import DataClass
from SBART.outside_tools.create_logger import setup_SBART_logger
from SBART.rv_calculation.RV_Bayesian.RV_Bayesian import RV_Bayesian
from SBART.rv_calculation.rv_stepping.RV_step import RV_step
from SBART.template_creation.StellarModel import StellarModel
from SBART.template_creation.TelluricModel import TelluricModel
from SBART.utils.custom_exceptions import InvalidConfiguration
from SBART.utils.spectral_conditions import (
    Empty_condition,
)


def config_update_with_fallback_to_default(
    config_dict, SBART_key_name, user_configs, user_key_name=None
):
    try:
        user_key_name = SBART_key_name if user_key_name is None else user_key_name
        config_dict[SBART_key_name] = user_configs[user_key_name]
    except KeyError:
        pass
    return config_dict


def run_target(
    rv_method,
    input_fpath,
    storage_path,
    instrument_name,
    user_configs,
    share_telluric=None,
    share_stellar=None,
    force_stellar_creation=False,
    force_telluric_creation=False,
    sampler_name=None,
    sampler_configs=None,
    log_to_terminal=False,
):
    for path in [share_telluric, share_stellar]:
        if path is not None and not os.path.exists(path):
            raise Exception("Trying to use a template that does not exist ({})".format(path))

    instrument = instrument_name_map[instrument_name]
    RVstep = user_configs["RVstep"]

    RV_limits = user_configs["RV_limits"]

    instrument_configs = user_configs.get("INSTRUMENT_CONFIGS", {})
    setup_SBART_logger(
        os.path.join(storage_path, "logs"),
        rv_method,
        instrument=instrument,
        log_to_terminal=log_to_terminal,
    )

    manager = DataClassManager()
    manager.start()

    data = manager.DataClass(
        input_fpath,
        storage_path=storage_path,
        instrument=instrument,
        instrument_options=instrument_configs,
        sigma_clip_RVs=user_configs.get("SIGMA_CLIP_RV", None),
    )

    if "REJECT_OBS" in user_configs:
        data.reject_observations(user_configs["REJECT_OBS"])

    data.generate_root_path(storage_path)

    interpol_properties = user_configs.get("INTERPOL_CONFIG_TEMPLATE", {})
    data.update_interpol_properties_of_all_frames(interpol_properties)

    inds = Indicators()
    data.remove_activity_lines(inds)

    telluric_model_configs = user_configs.get("TELLURIC_MODEL_CONFIGS", {})
    telluric_template_configs = user_configs.get("TELLURIC_TEMPLATE_CONFIGS", {})

    ModelTell = TelluricModel(
        usage_mode="individual",
        user_configs=telluric_model_configs,
        root_folder_path=storage_path if share_telluric is None else share_telluric,
    )

    ModelTell.Generate_Model(
        dataClass=data,
        telluric_configs=telluric_template_configs,
        force_computation=force_telluric_creation,
        store_templates=True,
    )
    data.remove_telluric_features(ModelTell)

    stellar_model_configs = user_configs.get("STELLAR_MODEL_CONFIGS", {})
    stellar_template_configs = user_configs.get("STELLAR_TEMPLATE_CONFIGS", {})

    ModelStell = StellarModel(
        user_configs=stellar_model_configs,
        root_folder_path=storage_path if share_stellar is None else share_stellar,
    )
    try:
        StellarTemplateConditions = user_configs["StellarTemplateConditions"]
    except KeyError:
        StellarTemplateConditions = Empty_condition()

    try:
        ModelStell.Generate_Model(
            data,
            stellar_template_configs,
            StellarTemplateConditions,
            force_computation=force_stellar_creation,
        )
        ModelStell.store_templates_to_disk(storage_path)
    except InvalidConfiguration:
        return
    data.ingest_StellarModel(ModelStell)

    interpol_properties = user_configs.get("INTERPOL_CONFIG_RV_EXTRACTION", {})
    data.update_interpol_properties_of_stellar_model(interpol_properties)

    confsRV = {"MEMORY_SAVE_MODE": stellar_template_configs["MEMORY_SAVE_MODE"]}

    confsRV = config_update_with_fallback_to_default(
        confsRV, "sigma_outliers_tolerance", user_configs
    )

    confsRV = {
        **confsRV,
        **user_configs.get("RV_Routine_extra_configs", {}),
    }

    if sampler_configs is None:
        sampler_configs = {}

    if sampler_name is not None:
        chosen_sampler = Sampler_map[sampler_name](user_configs=sampler_configs)
    else:
        if rv_method == "RV_step":
            sampler = Sampler_map["chi_squared"]
        elif rv_method in ["Laplace", "MCMC"]:
            sampler = Sampler_map[rv_method]
        else:
            raise Exception("Can't recognize method name!")
        chosen_sampler = sampler(RVstep, RV_limits, user_configs=sampler_configs)

    if rv_method == "RV_step":
        rv_model = RV_step(
            user_configs["NUMBER_WORKERS"],
            RV_configs=confsRV,
            sampler=chosen_sampler,
        )

        orders = user_configs["ORDER_SKIP"]
    else:
        confsRV = config_update_with_fallback_to_default(
            confsRV, "order_removal_mode", user_configs
        )
        rv_model = RV_Bayesian(
            math.ceil(user_configs["NUMBER_WORKERS"] / 2),
            RV_configs=confsRV,
            sampler=chosen_sampler,
        )
        orders = os.path.join(storage_path, "Iteration_0", "RV_step")

    rv_model.run_routine(data, storage_path, orders)

    # ensure that we dont reuse the logger
    setup_SBART_logger("", "", instrument=instrument, log_to_terminal=False, write_to_file=False)
