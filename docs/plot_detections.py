from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# 11.1868+0.0029−0.0031
# 1.07+0.06−0.06

# 5.122+0.002−0.036
# 0.26+0.05−0.05

# value, lower error, higher error
detections = [
    {
        "Instrument": "ESPRESSO",
        "Planet b": {
            "msini": (1.07, 0.06, 0.06),
            "P": (11.1868, 0.0031, 0.0029),
        },
        "Planet d": {
            "msini": (0.26, 0.05, 0.05),
            "P": (5.122, 0.036, 0.002),
        },
    },  # Faria+2022
    {
        "Instrument": "ESPRESSO",
        "Planet b": {"msini": (4.28, 0.33, 0.35), "P": (1.208974, 1e-6, 1e-6)},
        "Planet c": {"msini": (1.86, 0.39, 0.37), "P": (3.64810, 1e-5, 1e-5)},
        "Planet d": {"msini": (3.02, 0.57, 0.58), "P": (6.201812, 9e-6, 9e-6)},
    },  # Passegger+2024
    {
        "Instrument": "HARPS-N",
        "Planet b": {"msini": (8.3, 1.32, 1.32), "P": (6.527282, 0.000020, 0.000015)},
    },  # Palethorpe+2024
    {
        "Instrument": "ESPRESSO",
        "Planet b": {"msini": (3.40, 0.46, 0.46), "P": (1.2730991, 0.0000029, 0.0000029)},
        "Planet c": {"msini": (6.7, 1.1, 1.1), "P": (8.465651, 0.000035, 0.000035)},
    },  # Suárez Mascareño + 2024
    {
        "Instrument": "ESPRESSO",
        "Planet b": {"msini": (2.15, 0.17, 0.17), "P": (18.3140, 0.002, 0.002)},
        "Planet c": {"msini": (2.98, 0.29, 0.29), "P": (89.68, 0.10, 0.10)},
        "Planet d": {"msini": (5.82, 0.57, 0.57), "P": (647.6, 2.7, 2.5)},
    },  # Nari+2024
    {
        "Instrument": "ESPRESSO",
        "Planet b": {"msini": (0.37, 0.05, 0.05), "P": (3.1533, 0.0006, 0.0006)},
    },  # Nari+2024
    {
        "Instrument": "CARMENES",
        "Planet b": {"msini": (8.80, 0.76, 0.76), "P": (8.5, 0, 0)},
        "Planet c": {"msini": (12.4, 1.1, 1.1), "P": (29.7, 0, 0)},
    },  # Balsalobre Ruza + 2025
]
fig, axis = plt.subplots()


N = len(detections)
cmap = plt.get_cmap("viridis")

for index, paper_data in enumerate(detections):
    color = cmap(index / N)
    for key, detection in paper_data.items():
        if "planet" in key.lower():
            print(detection["P"][1:])
            axis.errorbar(
                detection["P"][0],
                detection["msini"][0],
                yerr=np.array(detection["msini"][1:])[:, np.newaxis],
                xerr=np.array(detection["P"][1:])[:, np.newaxis],
                marker="o",
                markersize=3,
                ls="",
                color=color,
            )
axis.set_ylim([0.1, 100])
axis.set_xlim([1, 1000])
axis.axhline(1, color="red", ls="--", alpha=0.2)
axis.axhline(10, color="red", ls="--", alpha=0.2)

axis.set_xscale("log")
axis.set_yscale("log")

axis.set_xlabel(r"Planet Period [days]")
axis.set_ylabel(r"$M_{p}\ sin(i)\ [M_{\bigoplus}$]")

fig.savefig(Path(__file__).parent / "sbart_detections.png")
