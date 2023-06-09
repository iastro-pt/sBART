from SBART.Instruments import ESPRESSO
from SBART.template_creation.TelluricModel import TelluricModel
from pathlib import Path

from SBART.outside_tools import create_logger

create_logger.setup_SBART_logger("", "", ESPRESSO, write_to_file=False)


from SBART.data_objects import DataClass, DataClassManager

data = DataClass(
    input_files=[Path("/home/amiguel/phd/spectra_collection/ESPRESSO/TauCeti/r.ESPRE.2022-07-06T09:23:45.925_S1D_A.fits")],
    instrument=ESPRESSO,
    storage_path="/home/amiguel/phd/tools/sBART_private/tests",
    instrument_options={}
)

ModelTell = TelluricModel(
    usage_mode="individual",
    user_configs={"CREATION_MODE": "telfit"},
    root_folder_path="/home/amiguel/phd/tools/sBART_private/tests/telfit_test"
)

ModelTell.Generate_Model(
    dataClass=data,
    telluric_configs={},
    force_computation=True,
    store_templates=True,
)