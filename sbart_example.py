import sys

from SBART.Instruments import ESPRESSO, HARPS
from pathlib import Path

current_folder = Path(__file__).parent.parent.absolute()

# FIle where each line is a disk path of a S2D file! Otherwise, list of files
input_filepath = ""

instrument = ESPRESSO

# Folder in which SBART will store its outputs
storage_path = current_folder / "to_delete"


from SBART.utils.units import meter_second

#rv_method = "classical"  # Either classical or Laplace or MCMC

for rv_method in ["classical", "Laplace"]:
    # Define the step that will be used for numerical calculations near max/min points
    RVstep = 0.1 * meter_second

    # Define the window, around the CCF RV, inside which the models can search for the optimal RV
    RV_limits = [200 * meter_second, 200 * meter_second]

    # List with orders to "throw" away
    orders_to_skip = list(range(48))

    # Number of cores to use
    N_cores = 10

    # For the S2D loading stage
    inst_options = {"minimum_order_SNR": 2,
                "apply_FluxCorr": True
                }

    # For the creation of the Telluric Model (i.e. the "template generator")
    telluric_model_configs = {"CREATION_MODE": "telfit"
                          }

    # For the creation of the individual Telluric templates
    telluric_template_genesis_configs = {
        "continuum_percentage_drop": 1
    }

    # For the creation of the Stellar Model (i.e. the "template generator")

    stellar_model_configs = {}

    # For the creation of the individual Stellar templates
    stellar_template_genesis_configs = {"MINIMUM_NUMBER_OBS": 2
                                    }

    confsRV = {"MEMORY_SAVE_MODE": False}

    from SBART.outside_tools.create_logger import setup_SBART_logger

    setup_SBART_logger(storage_path=storage_path / "logs", RV_method=rv_method, instrument=instrument, log_to_terminal=True)

    from SBART.data_objects import DataClassManager, DataClass

    manager = DataClassManager()
    manager.start()

    data = DataClass(
        input_filepath,
        storage_path=storage_path,
        instrument=instrument,
        instrument_options=inst_options,
    )

    from SBART.Quality_Control.activity_indicators import Indicators

    inds = Indicators()
    data.remove_activity_lines(inds)

    from SBART.template_creation.TelluricModel import TelluricModel

    ModelTell = TelluricModel(
        usage_mode="individual",
        user_configs=telluric_model_configs,
        root_folder_path=storage_path,
    )

    ModelTell.Generate_Model(dataClass=data,
                         telluric_configs=telluric_template_genesis_configs
                         )

    data.remove_telluric_features(ModelTell)

    from SBART.template_creation.StellarModel import StellarModel

    ModelStell = StellarModel(user_configs=stellar_model_configs,
                          root_folder_path=storage_path
                          )

    from SBART.utils.spectral_conditions import FNAME_condition, KEYWORD_condition

    StellarTemplateConditions = FNAME_condition(["r.ESPRE.2019-04-25T00:27:44.066_S2D_A.fits"]) + KEYWORD_condition("airmass", [[0, 1.5]])

    ModelStell.Generate_Model(data, stellar_template_genesis_configs, StellarTemplateConditions)

    ModelStell.store_templates_to_disk(storage_path)

    data.ingest_StellarModel(ModelStell)

    from SBART.rv_calculation.RV_Bayesian.RV_Bayesian import RV_Bayesian
    from SBART.rv_calculation.rv_stepping.RV_step import RV_step
    from SBART.Samplers import chi_squared_sampler, Laplace_approx, MCMC_sampler
    import os, emcee

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
            sampler = MCMC_sampler(RVstep, RV_limits, {"MAX_ITERATIONS": 1000,
                                                   "ensemble_moves": emcee.moves.GaussianMove(0.1)
                                                   }
                               )

        if rv_method == "Laplace":
            sampler = Laplace_approx(RVstep, RV_limits)

        rv_model = RV_Bayesian(
            processes=N_cores,
            RV_configs=confsRV,
            sampler=sampler,
        )
        orders = os.path.join(storage_path, "Iteration_0/RV_step")
    else:
        raise Exception

    rv_model.run_routine(data, storage_path, orders)
