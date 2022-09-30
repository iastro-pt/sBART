============================
FAQ
============================

SBART RVs have very different RVs than the CCF
===============================================

**Description:** In typical cases, the S-BART algorithm is less prone to numerical errors than the CCF. However, the classical algorithm might be affected by them, under some edge cases. In such cases, the uncertainty of a given spectral orders goes to very small values, having the larger weight in the (classical) RV estimate.

**Solution:** For such cases, check inside the *RV_step/individual_subInst/<instrument>/plots/* for the *RV_orderwise_errors.png* file, where we can find a plot of the RV uncertainty for each order. If we can clearly see a value much smaller than all others, rejecting that order might improve the overall solution.


SBART RVs have very large uncertainties
=========================================

**Description:** It is often the case that a single observation is leading to the rejection of a large number of spectral orders. Recall that we enforce that any given sub-Instrument uses exactly the same spectral orders.

**Solution:**  For such cases, check inside the *RV_step/individual_subInst/<instrument>/metrics/* for the *DataRejectionSummary.png* file, where we provide the reason behind the rejection of each spectral order of each observation. At the end of the file, there is also a short summary of the rejection.

