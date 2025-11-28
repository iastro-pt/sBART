import os
from pathlib import Path

import emcee
from ASTRA import setup_ASTRA_logger
from ASTRA.data_objects import DataClassManager
from ASTRA.Instruments import ESPRESSO
from ASTRA.Quality_Control.activity_indicators import Indicators
from ASTRA.template_creation.StellarModel import StellarModel
from ASTRA.template_creation.TelluricModel import TelluricModel
from ASTRA.utils.choices import TELLURIC_CREATION_MODE
from ASTRA.utils.spectral_conditions import FNAME_condition, KEYWORD_condition

from SBART import setup_SBART_logger
from SBART.rv_calculation.RV_Bayesian.RV_Bayesian import RV_Bayesian
from SBART.rv_calculation.rv_stepping.RV_step import RV_step
from SBART.Samplers import Laplace_approx, MCMC_sampler, chi_squared_sampler
from SBART.utils.units import meter_second

current_folder = Path(__file__).parent.parent.absolute()

# FIle where each line is a disk path of a S2D file! Otherwise, list of files
input_filepath = list(set(Path("/home/amiguel/Downloads/archive(2)").glob("**/*e2ds_A.fits")))
instrument = ESPRESSO

# Folder in which SBART will store its outputs
storage_path = Path("/tmp") / "to_delete"
storage_path.mkdir(exist_ok=True)


for rv_method in [
    "classical",
    "Laplace",
]:
    # Define the step that will be used for numerical calculations near max/min points
    RVstep = 0.1 * meter_second

    # Define the window, around the CCF RV, inside which the models can search for the optimal RV
    RV_limits = [200 * meter_second, 200 * meter_second]

    # List with orders to "throw" away
    orders_to_skip = list(range(60))
    # Number of cores to use
    N_cores = 10

    # For the S2D loading stage
    inst_options = {
        "minimum_order_SNR": 2,
        "apply_FluxCorr": True,
    }

    # For the creation of the Telluric Model (i.e. the "template generator")
    telluric_model_configs = {"CREATION_MODE": TELLURIC_CREATION_MODE.telfit}

    # For the creation of the individual Telluric templates
    telluric_template_genesis_configs = {"continuum_percentage_drop": 1}

    # For the creation of the Stellar Model (i.e. the "template generator")

    stellar_model_configs = {}

    # For the creation of the individual Stellar templates
    stellar_template_genesis_configs = {"MINIMUM_NUMBER_OBS": 2}

    confsRV = {"MEMORY_SAVE_MODE": False}

    setup_SBART_logger(
        log_path=storage_path / "logs",
        RV_method=rv_method,
        log_to_terminal=True,
    )

    setup_ASTRA_logger(
        storage_path=storage_path / "logs",
        log_to_terminal=True,
    )

    manager = DataClassManager()
    manager.start()

    data = manager.DataClass(
        input_filepath,
        storage_path=storage_path,
        instrument=instrument,
        instrument_options=inst_options,
    )

    inds = Indicators()
    data.remove_activity_lines(inds)

    ModelTell = TelluricModel(
        usage_mode="individual",
        user_configs=telluric_model_configs,
        root_folder_path=storage_path,
    )

    ModelTell.Generate_Model(dataClass=data, telluric_configs=telluric_template_genesis_configs)

    data.remove_telluric_features(ModelTell)

    ModelStell = StellarModel(user_configs=stellar_model_configs, root_folder_path=storage_path)

    StellarTemplateConditions = FNAME_condition(["r.ESPRE.2019-04-25T00:27:44.066_S2D_A.fits"]) + KEYWORD_condition(
        "airmass", [[0, 1.5]]
    )

    ModelStell.Generate_Model(data, stellar_template_genesis_configs, StellarTemplateConditions)

    ModelStell.store_templates_to_disk(storage_path)

    data.ingest_StellarModel(ModelStell)

    if rv_method == "classical":
        sampler = chi_squared_sampler(RVstep, RV_limits, user_configs={})
        rv_model = RV_step(
            processes=N_cores,
            RV_configs=confsRV,
            sampler=sampler,
        )

        orders = orders_to_skip
    elif rv_method in ["Laplace", "MCMC"]:
        if rv_method == "MCMC":
            sampler = MCMC_sampler(
                RVstep,
                RV_limits,
                {
                    "MAX_ITERATIONS": 1000,
                    "ensemble_moves": emcee.moves.GaussianMove(0.1),
                },
            )

        if rv_method == "Laplace":
            sampler = Laplace_approx(RVstep, RV_limits)

        rv_model = RV_Bayesian(
            processes=N_cores,
            RV_configs=confsRV,
            sampler=sampler,
        )
        orders = storage_path / "Iteration_0/RV_step"
    else:
        raise Exception

    rv_model.run_routine(data, storage_path, orders)
