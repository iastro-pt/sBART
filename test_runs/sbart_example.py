import sys

from SBART.Instruments import ESPRESSO, HARPS
from pathlib import Path

current_folder = Path(__file__).parent.parent.absolute()

# FIle where each line is a disk path of a S2D file! Otherwise, list of files
input_filepath = [
    "/home/amiguel/phd/tools/sBART_private/test_runs/data/r.ESPRE.2019-11-30T04:46:03.806_S2D_A.fits",
    "/home/amiguel/phd/tools/sBART_private/test_runs/data/r.ESPRE.2019-11-30T04:47:27.376_S2D_A.fits",
]

instrument = ESPRESSO

# Folder in which SBART will store its outputs
storage_path = current_folder / "test_runs" / "run_outputs"


from SBART.utils.units import meter_second

# rv_method = "classical"  # Either classical or Laplace or MCMC

for rv_method in ["classical", "Laplace"]:
    # Define the step that will be used for numerical calculations near max/min points
    RVstep = 0.1 * meter_second

    # Define the window, around the CCF RV, inside which the models can search for the optimal RV
    RV_limits = [300 * meter_second, 300 * meter_second]

    # List with orders to "throw" away
    orders_to_skip = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        98,
        99,
        100,
        101,
        110,
        111,
        116,
        117,
        118,
        119,
        120,
        121,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        164,
        165,
    ]

    # Number of cores to use
    N_cores = 10

    # For the S2D loading stage
    inst_options = {"minimum_order_SNR": 1.5, "apply_FluxCorr": True}

    # For the creation of the Telluric Model (i.e. the "template generator")
    telluric_model_configs = {"CREATION_MODE": "telfit", "EXTENSION_MODE": "window"}

    # For the creation of the individual Telluric templates
    telluric_template_genesis_configs = {"continuum_percentage_drop": 1}

    # For the creation of the Stellar Model (i.e. the "template generator")

    stellar_model_configs = {}

    # For the creation of the individual Stellar templates
    stellar_template_genesis_configs = {"MINIMUM_NUMBER_OBS": 2}

    confsRV = {"MEMORY_SAVE_MODE": False}

    from SBART.outside_tools.create_logger import setup_SBART_logger

    setup_SBART_logger(
        storage_path=storage_path / "logs",
        RV_method=rv_method,
        instrument=instrument,
        log_to_terminal=True,
    )

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

    ModelTell.Generate_Model(
        dataClass=data, telluric_configs=telluric_template_genesis_configs
    )

    data.remove_telluric_features(ModelTell)

    from SBART.template_creation.StellarModel import StellarModel

    ModelStell = StellarModel(
        user_configs=stellar_model_configs, root_folder_path=storage_path
    )

    from SBART.utils.spectral_conditions import FNAME_condition, KEYWORD_condition

    ModelStell.Generate_Model(data, stellar_template_genesis_configs)

    ModelStell.store_templates_to_disk(storage_path)

    data.ingest_StellarModel(ModelStell)

    from SBART.rv_calculation.RV_Bayesian.RV_Bayesian import RV_Bayesian
    from SBART.rv_calculation.rv_stepping.RV_step import RV_step
    from SBART.rv_calculation.ExpectedPrecision.RV_precision import RV_precision
    from SBART.Samplers import (
        chi_squared_sampler,
        Laplace_approx,
        MCMC_sampler,
        RVcontent_sampler,
    )

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
        orders = os.path.join(storage_path, "Iteration_0/RV_step")
    else:
        raise Exception

    rv_model.run_routine(data, storage_path, orders)
