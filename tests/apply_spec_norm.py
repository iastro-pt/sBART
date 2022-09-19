from SBART.Instruments import  ESPRESSO
from pathlib import Path
import matplotlib.pyplot as plt
main_path = Path("/home/amiguel/seminar/data_files/ESPRESSO/HD40307/HD40307_raw/data")
file_start = "r.ESPRE.2018-12-24T00:27:15.371_S2D_BLAZE_A.fits"
file_end = "r.ESPRE.2018-12-24T09:14:20.156_S2D_BLAZE_A.fits"


from SBART.outside_tools import create_logger
create_logger.setup_SBART_logger("","", ESPRESSO, write_to_file=False)
d = ESPRESSO(main_path / file_start, user_configs={"NORMALIZE_SPECTRA":True,
                                                   "NORMALIZATION_MODE":"Poly-Norm"})

d.generate_root_path("/home/amiguel/phd/tools/sBART_private/tests")

wave, flux, _ ,_ = d.get_data_from_spectral_order(100)
plt.plot(wave, flux)
d.normalize_spectra()
wave, flux, _ ,_ = d.get_data_from_spectral_order(100)
plt.plot(wave, flux)

d.trigger_data_storage()
plt.show()
