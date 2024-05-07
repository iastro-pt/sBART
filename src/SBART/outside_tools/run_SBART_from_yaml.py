import os
from copy import deepcopy
from pathlib import Path

import yaml

from SBART.outside_tools.run_SBART_from_config_dict import run_target
from SBART.utils.spectral_conditions import Empty_condition

from SBART.utils.units import meter_second, kilometer_second

from SBART.utils.spectral_conditions import (
    FNAME_condition,
    KEYWORD_condition,
    SubInstrument_condition,
    Empty_condition,
)


def run_SBART_from_yaml(target_config_file, main_run_output_path, only_run=()):
    configs = import_target_configs(target_config_file)

    for run_name, run_configs in configs:
        run_output_path = (
            main_run_output_path / run_configs.get("sub_group_name", "baseline_TM") / run_name
        )
        run_output_path.mkdir(exist_ok=True, parents=True)

        # Launch TM algorithm here !!!!!

        if len(only_run) != 0 and run_name not in only_run:
            print("Run name <{}> is disabled".format(run_name))
            continue

        multi_input_mode = run_configs.get("MULTI_INPUT_MODE", False)

        input_files = run_configs["DATA_FILE"].as_posix()

        if not os.path.exists(input_files) and len(run_configs["LOAD_ALL_FROM_FOLDER"]) == 0:
            print("Missing input text file for ")
            raise Exception

        if len(run_configs["LOAD_ALL_FROM_FOLDER"]) != 0:
            load_path = Path(run_configs["LOAD_ALL_FROM_FOLDER"])
            input_files = list(load_path.iterdir())
            input_files = [i.as_posix() for i in input_files]

        for RV_method in run_configs["RV_methods"]:
            final_storage_path = (
                run_output_path
                if not multi_input_mode
                else run_output_path / run_configs["DATA_FILE"].stem.split(".")[0]
            )
            final_storage_path.mkdir(exist_ok=True, parents=True)

            run_target(
                rv_method=RV_method,
                input_fpath=input_files,
                storage_path=final_storage_path,
                instrument_name=run_configs["INSTRUMENT"],
                user_configs=run_configs,
            )


def update_keyword_pair(config_dict, keyword, value):
    if keyword in ["StellarTemplateConditions", "RVstep", "RV_limits"]:
        # Worst idea ever.... But well, it works...
        config_dict[keyword] = eval(value)

    elif keyword in ["ExtraStellarTemplateConditions"]:
        config_dict["StellarTemplateConditions"] += eval(value)

    elif keyword in ["EXTRA_ORDERS_SKIP_RANGE", "ORDER_SKIP_RANGE"]:
        print("---", keyword)
        for item in value:
            print(list(range(item[0], item[1])))
            config_dict["ORDER_SKIP"].extend(range(item[0], item[1]))
    elif keyword == "EXTRA_ORDERS_SKIP":
        config_dict["ORDER_SKIP"].extend(value)
    else:
        config_dict[keyword] = value
    return config_dict


def handle_template_extra_conditions(
    current_config_dict, filename, multi_input_mode, modifier_dict
):
    general_key = "ExtraStellarTemplateConditions"
    try:
        current_config_dict = update_keyword_pair(
            current_config_dict, general_key, modifier_dict[general_key]
        )
        del current_config_dict[general_key]
    except KeyError:
        pass

    if multi_input_mode:
        input_name = filename.stem
        multi_input_key = "ExtraNightlyStellarTemplateConditions"
        extra_conditions = modifier_dict.get(multi_input_key, {})
        if input_name in extra_conditions:
            current_config_dict = update_keyword_pair(
                current_config_dict, general_key, extra_conditions[input_name]
            )

    return current_config_dict


def import_target_configs(config_path):
    loaded_configs = load_config(config_path)

    runs_definitions = []

    for run_name, run_configs in loaded_configs.items():
        if run_name in ["BASELINE_CONFIGS"]:
            continue

        actual_run_name = run_configs.get("RENAME_TO", run_name)

        Baseline_conf = deepcopy(loaded_configs["BASELINE_CONFIGS"])

        multi_input_mode = Baseline_conf.get("MULTI_INPUT_MODE", False)

        if multi_input_mode:
            data_to_run = Baseline_conf["DATA_FILE"]
        else:
            data_to_run = [Baseline_conf["DATA_FILE"]]

        overloaded_paths = False

        for file_path in data_to_run:
            if overloaded_paths:
                break
            run_conf = deepcopy(Baseline_conf)
            filename = Path(file_path)
            run_conf = update_keyword_pair(run_conf, "ORDER_SKIP", run_conf["ORDER_SKIP"])
            run_conf = update_keyword_pair(run_conf, "DATA_FILE", filename)

            try:
                run_conf = update_keyword_pair(
                    run_conf, "ORDER_SKIP_RANGE", run_conf["ORDER_SKIP_RANGE"]
                )
            except KeyError as e:
                pass

            try:
                for key in ["RVstep", "RV_limits"]:
                    run_conf = update_keyword_pair(run_conf, key, run_conf[key])
            except KeyError:
                pass

            try:
                key = "StellarTemplateConditions"
                run_conf = update_keyword_pair(run_conf, key, run_conf[key])
            except KeyError:
                run_conf["StellarTemplateConfigs"] = Empty_condition()

            run_conf = handle_template_extra_conditions(
                run_conf, filename, multi_input_mode, run_conf
            )
            if run_configs is not None:
                try:
                    key = "StellarTemplateConditions"
                    run_conf = update_keyword_pair(run_conf, key, run_configs[key])
                except KeyError:
                    pass

                try:
                    updated_data_file = run_configs["DATA_FILE"]
                    if not isinstance(updated_data_file, (str, Path)):
                        raise Exception("Multiple datafiles must be defined in the baseline!")
                    overloaded_paths = True
                    run_configs["DATA_FILE"] = Path(updated_data_file)
                except KeyError:
                    pass

                for param, value in run_configs.items():
                    if "TemplateConditions" in param:
                        continue

                    run_conf = update_keyword_pair(run_conf, param, value)

                run_conf = handle_template_extra_conditions(
                    run_conf, filename, multi_input_mode, run_configs
                )
            runs_definitions.append((actual_run_name, run_conf))

    return runs_definitions


def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    return configs
