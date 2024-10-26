"""Evaluate the evolution of the stellar template when the initial guess for the RVs is set at zero for every single observation

Questions to answer:

1) If we iterate enough, can we get back to reasonable rvs?
"""

from SBART.outside_tools.run_SBART_from_config_dict import run_target
from pathlib import Path

curr_folder = Path(__file__).parent.parent.absolute()

storage_folder_nominal = curr_folder / "benchmarks_results" / "template_iterations" / "nominal"

storage_folder_blind = curr_folder / "benchmarks_results" / "template_iterations" / "blindRVs"

data_to_test = ""


for const_rv_guess, storage_folder in [
    (True, storage_folder_blind),
    (False, storage_folder_nominal),
]:
    # Base run:
    rv_method = "RV_step"

    run_target(
        rv_method=rv_method,
        input_fpath=data_to_test,
        storage_path=storage_folder,
        instrument_name="ESPRESSO",
        user_configs={
            "STELLAR_TEMPLATE_CONFIGS": {
                "CONSTANT_RV_GUESS": const_rv_guess,
                "NUMBER_WORKERS": 5,
            }
        },
    )

    # Iterate on top of it
    for iteration in range(4):
        confs = {
            "STELLAR_MODEL_CONFIGS": {
                "ALIGNEMENT_RV_SOURCE": "SBART",
                "PREVIOUS_SBART_PATH": storage_folder / f"Iteration_{iteration}" / rv_method,
            }
        }

        run_target(
            rv_method=rv_method,
            input_fpath=data_to_test,
            storage_path=storage_folder,
            instrument_name="ESPRESSO",
            user_configs=confs,
            share_telluric=storage_folder,  # avoid creating multiple times the stellar template
        )
