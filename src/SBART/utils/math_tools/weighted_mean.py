import numpy as np


def weighted_mean(orders_RV, squared_uncertainties, RV_variance_estimator="simple"):
    finite_rvs = np.isfinite(orders_RV)
    finite_errs = np.isfinite(squared_uncertainties)

    if not np.equal(finite_errs, finite_rvs).all():
        raise Exception("Different position of finite elements on the weighted mean!")

    weights = np.divide(1, squared_uncertainties)  # 1/ e**2
    sum_weights = np.nansum(weights, axis=1)
    final_RV = np.nansum(np.multiply(weights, orders_RV), axis=1) / sum_weights

    No = squared_uncertainties.shape[1] - np.sum(np.isnan(squared_uncertainties), axis=1)
    if RV_variance_estimator == "simple":
        final_error = np.sqrt(1 / (sum_weights))
    elif RV_variance_estimator == "with_correction":
        final_error = np.sqrt(
            np.nansum(weights * (orders_RV - final_RV[:, np.newaxis]) ** 2, axis=1)
            / (sum_weights * (No - 1))
        )

    return final_RV, final_error


if __name__ == "__main__":
    orders = np.array([[1, 1, 2, np.nan], [0.1, 2, 3, 1]])
    errors = np.ones_like(orders)
    errors[0, -1] = np.nan
    print(weighted_mean(orders, errors), np.nanmean(orders, axis=1))
