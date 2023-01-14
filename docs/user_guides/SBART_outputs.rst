================================
File outputs of SBART
================================

By default, SBART will store the majority of its outputs to disk, allowing one to:

i) Understand the root cause behind possible numerical inconsistencies in the final RVs
ii) Allow SBART to easily load the outputs of a previous run

Each interface will thus handle the disk storage of its own data products, creating independent folders for such purpose.

Organization of the  data products
======================================================

For most end-users, the more interesting data products will live in the folders whose name is the same as the RV routine that was selected (e.g. RV_step; S-BART). Inside those folders you might find other folders, which will be named in accordance with the selected Sampler (e.g. MCMC, Laplace). They are structured identically.

The high level data products will be divided, accordingly to the methodology used to select the spectral orders to be rejected. This division is made in two folders:

- individual_subInst - Results that do **not** share rejections between different sub-Instruments
- merged_subInst - Results that come from considering the set of orders that is common to all sub-Instruments

Inside the two folders we can find exactly the same file structure, which contains:

- A folder for each sub-Instrument of the current instrument
- A .txt file that contains the RV time-series for all Frames. This file format follows the structure of `The output RV file`_.


The output RV file
-------------------------------

The RV file is a text file with multiple columns. A brief description of them follows here:

- BJD:  BJD of the observation, as taken from the header of the S2D files
- RVc [Km/s]: RV measurement, in km/s, after being corrected by the instrumental drift (see DRIFT column description) and the secular acceleration of the star (see SA column description)
- RVc_ERR [Km/s]:  RV uncertainty
- SA [m/s]: Value of the secular acceleration, assuming that same "time origin" as DACE. If, for some reason, the values are all at zero, then it means that sBART was not able of querying DACE for the given star and, consequently, there was no SA correction. To obtain the "raw" sBART RVs, simply sum this column to the RVc column
- DRIFT [m/s]: Instrumental drift that is used to correct RVs. If the values are "nan", then it means that the drift correction was already made at the DRS level
- DRIFT_ERR: Drift uncertainty. If not "nan", then it is added in quadrature to the "raw" sBART RVs
- filename: Filename of this observation
- frameIDs: ID of the observation inside sBART, only used for debugging / accessing extra information in sBART outputs
- DLW and DLW_ERR: Implementation of the DLW algorithm, still under testing to understand if it is working as expected

Data products from the creation of the templates
======================================================

Aside from the RV data, SBART also stores the data from both stellar and telluric templates. This is stored inside the *templates* folder and, at a later date, SBART can load the data from disk, allowing one to use the object interface of SBART.


The Stellar Template
-----------------------

The Stellar template stores to disk a large number of files and folders:

- data_products - internal folder to store information, no use-case for the end-user
- metrics - Store plots with the amount of pixels rejected in each spectral order. Allows to diagnose possible chromatic issues (across multiple observations) or to detect outlier observations

- The *fits* files will store the actual template, which can then be loaded with SBART
- The text files store an overview of the data included (and rejected) from the stellar template. It also provided a summary of the configs and the data conditions that were applied


The Telluric Template
-----------------------

The telluric template stores to disk:

i) One fits file for each sub-Instrument, with the binary mask and the transmittance spectra
ii) One "metrics" folder, where there are plots of the transmittance spectra used for the "basis" of the telluric mask

