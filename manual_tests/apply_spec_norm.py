from SBART.Instruments import ESPRESSO
from pathlib import Path
import matplotlib.pyplot as plt

main_path = Path("/home/amiguel/seminar/data_files/ESPRESSO/HD40307/HD40307_raw/data")
file_start = "r.ESPRE.2018-12-24T00:27:15.371_S2D_BLAZE_A.fits"
file_end = "r.ESPRE.2018-12-24T09:14:20.156_S2D_BLAZE_A.fits"

if 1:
    main_path = Path("/home/amiguel/phd/spectra_collection/ESPRESSO/Proxima")
    file_start = "r.ESPRE.2021-04-14T01:53:56.407_S2D_BLAZE_A.fits"

    # main_path = Path("/home/amiguel/seminar/data_files/ESPRESSO/TauCet/data")
    # file_start = "r.ESPRE.2021-10-03T08:39:41.550_S2D_BLAZE_A.fits"

from astropy.io import fits
with fits.open(main_path/file_start) as hdu:
    sci = hdu[1].data
    wave = hdu[4].data

from SBART.outside_tools import create_logger

create_logger.setup_SBART_logger("", "", ESPRESSO, write_to_file=False)

d = ESPRESSO(main_path / file_start, user_configs={"NORMALIZE_SPECTRA": False,
                                                   "NORMALIZATION_MODE": "Poly-Norm",
                                                   "apply_FluxBalance_Norm":False,
                                                   "apply_FluxCorr":False
                                                   }
             )

d.generate_root_path("/home/amiguel/phd/tools/sBART_private/tests")

fig, axis = plt.subplots(2,1, sharex=True)
wave, flux, _, _ = d.get_data_from_spectral_order(150)
axis[0].plot(wave, flux)
# d.normalize_spectra()
wave, flux, _, _ = d.get_data_from_spectral_order(150)
axis[1].plot(wave, flux)

axis[1].set_xlabel("Wavelength")

axis[0].set_ylabel("Flux")
axis[1].set_ylabel("Flux")
# d.trigger_data_storage()
plt.show()
