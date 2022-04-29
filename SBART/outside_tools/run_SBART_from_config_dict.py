import math
import os

from SBART.Instruments import ESPRESSO, HARPS
from SBART.Quality_Control.activity_indicators import Indicators
from SBART.Samplers import chi_squared_sampler, Laplace_approx
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


def run_target(rv_method, input_fpath, storage_path, instrument_name, user_configs):
    instrument_name_map = {"ESPRESSO": ESPRESSO, "HARPS": HARPS}

    instrument = instrument_name_map[instrument_name]
    RVstep = user_configs["RVstep"]

    RV_limits = user_configs["RV_limits"]

    inst_options = {}

    if "minimum_order_SNR" in user_configs:
        inst_options = config_update_with_fallback_to_default(
            inst_options, "minimum_order_SNR", user_configs
        )
    if instrument_name_map == "KOBE":
        inst_options["shaq_output_folder"] = user_configs["KOBE_SHAQ_FOLDER_PATH"]

    if instrument_name_map == "ESPRESSO":
        inst_options = config_update_with_fallback_to_default(
            inst_options, "apply_FluxCorr", user_configs
        )

    if "inst_extra_configs" in user_configs:
        inst_options = {**inst_options, **user_configs["inst_extra_configs"]}

    setup_SBART_logger(
        os.path.join(storage_path, "logs"),
        rv_method,
        instrument=instrument,
        log_to_terminal=True,
    )

    manager = DataClassManager()
    manager.start()

    data = manager.DataClass(
        input_fpath,
        instrument=instrument,
        instrument_options=inst_options,
    )

    inds = Indicators()
    data.remove_activity_lines(inds)

    if rv_method == "RV_step" and 0:
        force_stellar_computation = True
    else:
        force_stellar_computation = False

    telluric_model_configs = {}

    telluric_model_configs = config_update_with_fallback_to_default(
        telluric_model_configs, "CREATION_MODE", user_configs, "TELLURIC_CREATION_MODE"
    )
    telluric_model_configs = config_update_with_fallback_to_default(
        telluric_model_configs, "EXTENSION_MODE", user_configs, "TELLURIC_EXTENSION_MODE"
    )
    telluric_template_genesis_configs = {
        "atmosphere_profile": "download",
        "download_tapas": True,
    }

    telluric_template_genesis_configs = config_update_with_fallback_to_default(
        telluric_template_genesis_configs, "user_info", user_configs
    )

    telluric_template_genesis_configs = config_update_with_fallback_to_default(
        telluric_template_genesis_configs, "download_path", user_configs, "TAPAS_download_path"
    )

    telluric_template_genesis_configs = config_update_with_fallback_to_default(
        telluric_template_genesis_configs,
        "continuum_percentage_drop",
        user_configs,
        "TELLURIC_continuum_percentage_drop",
    )

    ModelTell = TelluricModel(
        usage_mode="individual",
        user_configs=telluric_model_configs,
        root_folder_path=storage_path,
    )

    ModelTell.Generate_Model(
        dataClass=data,
        telluric_configs=telluric_template_genesis_configs,
        force_computation=False,
        store_templates=True,
    )
    data.remove_telluric_features(ModelTell)

    stellar_model_configs = {}

    stellar_model_configs = config_update_with_fallback_to_default(
        stellar_model_configs, "CREATION_MODE", user_configs, "STELLAR_CREATION_MODE"
    )

    ModelStell = StellarModel(user_configs=stellar_model_configs, root_folder_path=storage_path)

    try:
        StellarTemplateConditions = user_configs["StellarTemplateConditions"]
    except KeyError:
        StellarTemplateConditions = Empty_condition()

    stellar_template_genesis_configs = {
        "MEMORY_SAVE_MODE": user_configs.get("MEMORY_SAVING_MODE", True),
        "NUMBER_WORKERS": (user_configs.get("NUMBER_WORKERS", 8), 1),
    }

    stellar_template_genesis_configs = {
        **stellar_template_genesis_configs,
        **user_configs.get("StellarTemplate_extra_configs", {}),
    }

    stellar_template_genesis_configs = config_update_with_fallback_to_default(
        stellar_template_genesis_configs, "CREATION_MODE", user_configs, "STELLAR_CREATION_MODE"
    )

    try:
        ModelStell.Generate_Model(
            data,
            stellar_template_genesis_configs,
            StellarTemplateConditions,
            force_computation=force_stellar_computation,
        )
        ModelStell.store_templates_to_disk(storage_path)
    except InvalidConfiguration:
        return

    data.ingest_StellarModel(ModelStell)

    confsRV = {"MEMORY_SAVE_MODE": stellar_template_genesis_configs["MEMORY_SAVE_MODE"]}

    confsRV = config_update_with_fallback_to_default(
        confsRV, "sigma_outliers_tolerance", user_configs
    )

    confsRV = {
        **confsRV,
        **user_configs.get("RV_Routine_extra_configs", {}),
    }

    if rv_method == "RV_step":
        sampler = chi_squared_sampler(RVstep, RV_limits)
        rv_model = RV_step(
            stellar_template_genesis_configs["NUMBER_WORKERS"][0],
            1,
            RV_configs=confsRV,
            sampler=sampler,
        )

        orders = user_configs["ORDER_SKIP"]
    else:
        confsRV = config_update_with_fallback_to_default(
            confsRV, "order_removal_mode", user_configs
        )
        sampler = Laplace_approx(RVstep, RV_limits)
        rv_model = RV_Bayesian(
            math.ceil(stellar_template_genesis_configs["NUMBER_WORKERS"][0] / 2),
            3,
            RV_configs=confsRV,
            sampler=sampler,
        )
        orders = os.path.join(storage_path, "RV_step")

    rv_model.run_routine(data, storage_path, orders)

    # ensure that we dont reuse the logger
    setup_SBART_logger("", "", instrument=instrument, log_to_terminal=False, write_to_file=False)
