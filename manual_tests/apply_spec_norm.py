from SBART.Instruments import ESPRESSO
from SBART.template_creation.TelluricModel import TelluricModel

from pathlib import Path
import matplotlib.pyplot as plt

main_path = Path("/home/amiguel/seminar/data_files/ESPRESSO/HD40307/HD40307_raw/data")
file_start = "r.ESPRE.2018-12-24T00:27:15.371_S2D_A.fits"
file_end = "r.ESPRE.2018-12-24T09:14:20.156_S2D_BLAZE_A.fits"

if 0:
    main_path = Path("/home/amiguel/phd/spectra_collection/ESPRESSO/Proxima")
    file_start = "r.ESPRE.2021-04-14T01:53:56.407_S2D_BLAZE_A.fits"

    # main_path = Path("/home/amiguel/seminar/data_files/ESPRESSO/TauCet/data")
    # file_start = "r.ESPRE.2021-10-03T08:39:41.550_S2D_BLAZE_A.fits"

from SBART.outside_tools import create_logger

create_logger.setup_SBART_logger("", "", ESPRESSO, write_to_file=False)

plt.switch_backend('TKagg')

from SBART.data_objects import DataClass, DataClassManager

data = DataClass(
    input_files=[Path("/home/amiguel/phd/spectra_collection/ESPRESSO/TauCeti/r.ESPRE.2022-07-06T09:23:45.925_S1D_A.fits")],
    instrument=ESPRESSO,
    storage_path="/home/amiguel/phd/tools/sBART_private/tests",
    instrument_options={"NORMALIZE_SPECTRA": True,
                        "NORMALIZATION_MODE": "RASSINE",
                        "RASSINE_path": Path("/home/amiguel/phd/tools/Rassine_modified"),
                        }
    )
data.normalize_all()

data.trigger_data_storage()
data.replace_frames_with_S2D_version()

ModelTell = TelluricModel(
    usage_mode="individual",
    user_configs={"CREATION_MODE": "telfit"},
    root_folder_path="/home/amiguel/phd/tools/sBART_private/tests"
)

ModelTell.Generate_Model(
    dataClass=data,
    telluric_configs={},
    force_computation=False,
    store_templates=True,
)
data.remove_telluric_features(ModelTell)

d = data.get_frame_by_ID(0)
d.build_mask(bypass_QualCheck=True)
fig, axis = plt.subplots(1, 1)
fig, axis1 = plt.subplots(1, 1)
axis = [axis]

for order in range(5, 7):
    wave, flux, _, mask = d.get_data_from_spectral_order(order, include_invalid=True)
    axis[0].plot(wave[~mask], flux[~mask], color="black")
    axis1.scatter(wave, mask, color="black")
    axis[0].plot(wave[mask], flux[mask] * 1.1, color="red", alpha=0.3)
    # d.normalize_spectra()

plt.show()
