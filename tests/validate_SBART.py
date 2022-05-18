from SBART.outside_tools import run_SBART_from_yaml


from pathlib import Path

path = Path(__file__).parent.absolute()

yaml_path = path / "teste_configs.yaml"
storage_path = path / "model_outputs"

storage_path.mkdir(parents=True, exist_ok=True)


run_SBART_from_yaml.run_SBART_from_yaml(target_config_file=yaml_path,
                                        main_run_output_path=storage_path,
                                        only_run=()
                                        )