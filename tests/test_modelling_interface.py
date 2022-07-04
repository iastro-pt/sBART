from pathlib import Path

import numpy as np

curr_folder = Path(__file__).absolute().parent
import sys
sys.path.insert(0, "/home/amiguel/work/SBART_GP_interpolation/sBART_private")

from SBART.Instruments import ESPRESSO
from SBART.outside_tools.create_logger import setup_SBART_logger
from SBART.utils.shift_spectra import interpolate_data, apply_RVshift
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

setup_SBART_logger(
    "",
    "",
    instrument=ESPRESSO,
    log_to_terminal=True,
    write_to_file=False
)

in_data = ["/data/work/jfaria/TM-nightly/HD40307_DACE_downloads/r.ESPRE.2018-12-24T00:45:21.184_S2D_A.fits"]

frames = []

for kern in ["Matern-3_2", "Matern-5_2"]:
    f_path = ESPRESSO(file_path=in_data[0],
                      user_configs={"GP_KERNEL": kern}
                      )

    f_path.generate_root_path(curr_folder / "GP_fit")
    frames.append(f_path)

order = 100
wave, flux, uncerts, mask = frames[0].get_data_from_spectral_order(order=order)

# for frame in frames:
#     frame.generate_model_from_order(order=order)
#
#     frame.trigger_data_storage()
#
#
XX = apply_RVshift(wave, stellar_RV=100 / 1000)

#
# new_spec, new_errors, indexes = interpolate_data(original_lambda=wave,
#                                                  original_spectrum=flux,
#                                                  original_errors=uncerts,
#                                                  new_lambda=XX,
#                                                  propagate_interpol_errors="interpolation",
#                                                  lower_limit=-np.inf,
#                                                  upper_limit=np.inf
#                                                  )

fig, axis = plt.subplots(2, 1)

# colors = ["red", "green", "yellow"]

# for ind, mode in enumerate(["cubic", "quadratic", "pchip"]):
#     frames[0].set_interpolation_properties({"SPLINE_TYPE": mode})
#     new_spec, new_errs = frames[0].interpolate_spectrum_to_wavelength(order=order, new_wavelengths=XX, shift_RV_by=0, RV_shift_mode="apply")
#     axis[0].plot(XX, new_spec, color=colors[ind], linestyle="-.")

frames[0].set_interpolation_properties({"INTERPOL_MODE": "NN"})

import time 
t = time.time()
new_spec, new_errs = frames[0].interpolate_spectrum_to_wavelength(order=order, new_wavelengths=XX, shift_RV_by=0,
                                                                  RV_shift_mode="apply"
                                                                  )
print(time.time() - t)
axis[0].plot(XX, new_spec, color="orange", linestyle="-.")
frames[0].trigger_data_storage()
# for f_index, frame in enumerate(frames):
    # mean, mu = f_path.interpolate_spectrum_to_wavelength(new_wavelengths=XX, order=order)
    # axis[0].plot(XX, mean, color=colors[f_index], linestyle="--")
    # axis[0].fill_between(XX, mean - mu, mean + mu, color=colors[f_index], alpha=0.3)
#     axis[1].scatter(XX, mean / new_spec, color=colors[f_index])
#
#
# axis[0].fill_between(XX, new_spec - new_errors, new_spec + new_errors, color="blue", alpha=0.3)
#
inds = np.where(np.logical_and(wave >= XX[0], wave <= XX[-1]))
axis[0].plot(wave[inds], flux[inds], color="black")
plt.show()
