import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import median_abs_deviation

from SBART.utils.RV_utilities.continuum_fit import fit_continuum_level
from SBART.utils.RV_utilities.create_spectral_blocks import build_blocks
from SBART.utils.shift_spectra import apply_RVshift, interpolate_data


def compute_outliers(
    obs_rv,
    spectra_wavelengths,
    spectra,
    spectra_mask,
    template_wavelengths,
    template,
    template_mask,
    worker_configs,
    temp_uncert,
    spec_uncert,
    epoch=None,
    order=None,
):
    """
    Find outliers for the spectra of one order of one observation!
    Parameters
    =====================
    obs_rv: float
        RV of the given observation
    spectra_wavelengths:
    spectra:
    spectra_mask:
        mask with ones in places where the data is valid
    template_wavelengths:
    template:
    template_mask:
        mask with ones in places where the data is valid
    tell_tolerance: float
        To be multiplied by the median difference to use as a threshold. If equal to or below zero, nothing is done

    Returns
    ========================
    new_mask : np.ndarray
        Points to be considered, i.e. mask with 1 in places to keep. It is an extension of the spectra_mask input array!
    found: bool
        Convergence achieved
    iterations:  int
        Number of iterations needed for convergence
    """
    tell_tolerance = worker_configs["OUTLIER_TOLERANCE"]

    if tell_tolerance <= 0:
        return spectra_mask, False, -1

    max_iter = worker_configs["MAX_ITERATIONS"]

    number_iterations = 0

    temp_mask = template_mask
    temp_order = template[temp_mask]
    temp_uncert_order = temp_uncert[temp_mask]
    temp_wave = template_wavelengths[temp_mask]

    new_mask = np.ones(spectra_mask.shape, dtype=np.bool)
    full_outlier_mask = np.ones(spectra_mask.shape, dtype=np.bool)

    # SHift template to previousRV of the star. SHould be close enough to flag large outliers
    shifted_tell_waves = apply_RVshift(temp_wave, obs_rv)

    # Find the template blocks and the (shifted) wavelength at the start and end of each!
    template_blocks = build_blocks(np.where(temp_mask == 1))
    blocks = []
    for block in template_blocks:  # account for gaps in the template
        start = apply_RVshift(template_wavelengths[block[0]], obs_rv)
        end = apply_RVshift(template_wavelengths[block[-1]], obs_rv)
        blocks.append((start, end))

    found = False
    for iterations in range(max_iter):
        # reseting the mask to be full of ones
        # the new_mask array will have 1 in places to keep and zeros in place to remove!
        new_mask = full_outlier_mask

        current_mask = np.logical_and(
            new_mask, spectra_mask
        )  # both are masks with 1 in places to keep

        # only consider the current "valid" set of spectral points
        spectra_flux = spectra[current_mask]
        spectra_wave = spectra_wavelengths[current_mask]
        spectra_uncert = spec_uncert[current_mask]

        interpolate_wave_indexes = np.zeros(spectra_wave.shape, dtype=np.bool)
        for wavelengths_block in blocks:
            # calculates the common wavelengths, for all RV shifts
            # first value: highest initial wavelenngth
            # last_value: smallest final wavelength
            wavelengths_limits = np.where(
                np.logical_and(
                    spectra_wave >= wavelengths_block[0],
                    spectra_wave <= wavelengths_block[1],
                )
            )
            interpolate_wave_indexes[wavelengths_limits] = True
        new_template, interpol_errors, indexes = interpolate_data(
            original_lambda=shifted_tell_waves,
            original_spectrum=temp_order,
            original_errors=temp_uncert_order,
            new_lambda=spectra_wave[interpolate_wave_indexes],
            lower_limit=0,
            upper_limit=np.inf,
            propagate_interpol_errors="interpolation",
            interpol_cores=0,
        )

        coefs, _, _, chosen_trend = fit_continuum_level(
            spectra_wave,
            spectra_flux[interpolate_wave_indexes],
            new_template,
            interpolate_wave_indexes,
            fit_degree=worker_configs["CONTINUUM_FIT_POLY_DEGREE"],
        )
        normalizer = chosen_trend(spectra_wave[interpolate_wave_indexes], *coefs)

        new_template = new_template * normalizer

        ### COmputing the outlier metric
        # calculating an offset to avoid division by zero

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation
        # https://stats.stackexchange.com/questions/123895/mad-formula-for-outlier-detection

        if worker_configs["METRIC_TO_USE"] == "MAD":
            metric = (spectra_flux[interpolate_wave_indexes] - new_template) / np.sqrt(
                interpol_errors ** 2 + spectra_uncert[interpolate_wave_indexes] ** 2
            )

            metric = (metric - np.median(metric)) / median_abs_deviation(metric, scale="normal")
            threshold = tell_tolerance
            # np.in1d returns a boolean array with the locations of the selected points
            mismatch_full_point = np.where(
                np.in1d(
                    spectra_wavelengths,
                    spectra_wave[interpolate_wave_indexes][np.where(np.abs(metric) >= threshold)],
                )
            )
        else:
            metric = np.log(np.abs(spectra_flux[interpolate_wave_indexes] / new_template))
            median = np.median(metric)
            threshold = tell_tolerance * np.std(metric)
            # np.in1d returns a boolean array with the locations of the selected points
            mismatch_full_point = np.where(
                np.in1d(
                    spectra_wavelengths,
                    spectra_wave[interpolate_wave_indexes][
                        np.where(
                            np.logical_or(
                                metric >= median + threshold, metric <= median - threshold
                            )
                        )
                    ],
                )
            )

        if len(mismatch_full_point[0]) == 0:
            # no points where selected to be removed!
            found = True
            break

        # sometimes we gut stuck on a infinite loop. storing previous 2 mismatches array to avoid it. MIght still be stuck in this loop
        if iterations > 2:
            if np.array_equal(mismatch_full_point, previous) or np.array_equal(
                mismatch_full_point, two_ago
            ):
                found = True
                break
        if iterations > 1:
            two_ago = previous
        previous = mismatch_full_point

        number_iterations += 1
        full_outlier_mask[previous] = False

    # places to keep are stored as 1
    updated_mask = np.logical_and(full_outlier_mask, spectra_mask)
    if not found:
        print(f" failed in {max_iter} iters. Using flagged points from last iteration")

    if epoch == 0 and order in [100] and 0:
        # Barnard after
        plt.rcParams.update({"font.size": 14})
        print("here", number_iterations)
        fig, axis = plt.subplots(2, 1, sharex=True, sharey="row", figsize=(14, 3))
        inds = np.where(full_outlier_mask == False)
        axis[0].plot(spectra_wave, spectra_flux, color="black", label="Spectra")
        axis[0].plot(
            spectra_wave[interpolate_wave_indexes],
            new_template,
            color="red",
            label="Normalized template",
        )
        axis[0].scatter(
            spectra_wavelengths[inds],
            spectra[inds],
            color="blue",
            label="masked points",
        )

        ax = axis[1]

        ax.scatter(
            spectra_wave[interpolate_wave_indexes],
            metric,
            color="black",
            label="Relative differences",
        )
        median = 0
        ax.axhline(median, color="red", label="Threshold")

        ax.axhline(median + threshold, color="red", linestyle="--")
        ax.axhline(median - threshold, color="red", linestyle="--")

        [axis[i].legend(ncol=3) for i in [0, 1]]

        axis[1].set_xlabel(r"$\lambda [\AA]$")
        axis[0].set_ylabel("Spectra")
        axis[1].set_ylabel("Removal metric")

        import matplotlib

        class OOMFormatter(matplotlib.ticker.ScalarFormatter):
            def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=False):
                self.oom = order
                self.fformat = fformat
                matplotlib.ticker.ScalarFormatter.__init__(
                    self, useOffset=offset, useMathText=mathText
                )

            def _set_order_of_magnitude(self):
                self.orderOfMagnitude = self.oom

            def _set_format(self, vmin=None, vmax=None):
                self.format = self.fformat
                if self._useMathText:
                    self.format = r"$\mathdefault{%s}$" % self.format

        fig, axis = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
        # axis[1,0].get_shared_y_axes().join(axis[1,0], axis[1,1])

        for y_dir, waves in enumerate([[6091, 6094], [6103, 6105]]):
            # axis[0, y_dir].set_title(order)
            norm = np.mean(
                spectra_flux[
                    np.where(
                        np.logical_and(spectra_wave > waves[0] - 0.2, spectra_wave < waves[1] + 0.2)
                    )
                ]
            )
            axis[0, y_dir].plot(spectra_wave, spectra_flux / norm, color="black", label="Spectra")
            axis[0, y_dir].plot(
                spectra_wave[interpolate_wave_indexes],
                new_template / norm,
                color="red",
                label="Normalized template",
            )
            axis[0, y_dir].scatter(
                spectra_wavelengths[inds],
                spectra[inds] / norm,
                color="blue",
                marker="x",
                label="masked points",
            )

            axis[1, y_dir].plot(
                spectra_wave[interpolate_wave_indexes],
                metric,
                color="black",
                label="Metric",
                linestyle="--",
                marker="o",
            )

            axis[1, y_dir].axhline(
                median + threshold, color="red", linestyle="dotted", label="Threshold"
            )
            axis[1, y_dir].axhline(median - threshold, color="red", linestyle="dotted")

            axis[1, y_dir].axhline(median, color="deepskyblue", label="Median")

            for i in range(2):
                if 1:
                    if y_dir == 0:
                        ticks = np.arange(waves[0], waves[1] + 0.1, 0.2)
                        axis[i, y_dir].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                        axis[i, y_dir].set_xticklabels(
                            [f"{i:.1f}" for i in ticks[:-2]] + [f"{ticks[-1]:.0f}"]
                        )

                    else:
                        ticks = np.arange(waves[0], waves[1] + 1, 1)
                        print(ticks)
                        axis[i, y_dir].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                        # axis[i,y_dir].set_xticklabels(ticks)
                        axis[i, y_dir].set_xticks(ticks)

                axis[i, y_dir].set_xlim(waves)

                # axis[i,y_dir].set_xticks(ticks)
        label_fsize = 15

        for y_dir in [0, 1]:
            axis[1, y_dir].set_xlabel(r"$\lambda[\AA]$", fontsize=label_fsize)
            axis[0, y_dir].set_yticklabels([])
            axis[0, y_dir].set_yticks([], minor=True)
            axis[0, y_dir].set_yticks([])
        axis[0, 0].set_ylim([0.55, 1.4])
        axis[0, 0].set_ylabel("Normalized Flux", fontsize=label_fsize, labelpad=28)
        axis[1, 0].set_ylabel("Metric", fontsize=label_fsize)

        axis[0, 0].legend(bbox_to_anchor=(-0.02, 1.08), ncol=4, loc="lower left")
        axis[1, 0].legend(bbox_to_anchor=(-0.02, 1.01), ncol=4, loc="lower left")

        if 0:
            axis[0, 1].yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
            axis[0, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axis[0, 0].set_ylim([2500, 5500])
            axis[0, 0].set_yticks([2500, 3500, 4500, 5500])

            # axis[1,1].set_yticklabels([])
            axis[0, 1].set_ylim([1000, 2500])
            axis[0, 1].set_yticks([1000, 1500, 2000, 2500])

        plt.tight_layout()
        axis[0, 0].set_title("Center of order", loc="left")
        axis[0, 1].set_title("Edge of order", loc="left")

        plt.subplots_adjust(hspace=0.25, wspace=0.165, top=0.9, right=0.98)

        if False:
            plt.savefig("/home/amiguel/work/to_download/template_outliers.pdf")

        plt.show()
    return updated_mask, found, number_iterations
