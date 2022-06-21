from SBART.Instruments import ESPRESSO

from SBART.data_objects import DataClass
from SBART.template_creation.TelluricModel import TelluricModel
from pathlib import Path
from SBART.outside_tools import create_logger
from pathlib import Path


create_logger.setup_SBART_logger(storage_path="", write_to_file=False, instrument=ESPRESSO,RV_method="")
storage_path = Path(__file__).absolute().parent / "telfit_fitting_tests"

# in_data = [i.as_posix() for i in (Path(__file__).absolute().parent / "resources").iterdir()]

# in_data = ["/home/amiguel/seminar/data_files/ESPRESSO/TauCet/data/r.ESPRE.2021-10-03T08:42:40.300_S2D_BLAZE_A.fits"]
in_data = ["/home/amiguel/seminar/data_files/ESPRESSO/proxima/0/r.ESPRE.2019-02-10T08:25:37.895_S2D_A.fits"]

data = DataClass(instrument=ESPRESSO,
                 path=in_data,
                 instrument_options={}
                 )

tell_conf = {"CREATION_MODE": "telfit",
             "APPLICATION_MODE": "correction"
             }

Model = TelluricModel(usage_mode="individual",
                      user_configs=tell_conf,
                      root_folder_path=storage_path
                      )

Model.Generate_Model(dataClass=data,
                     telluric_configs={},

                     )
data.remove_telluric_features(Model)