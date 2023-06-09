=============
Release Notes
=============

.. towncrier release notes start

SBART 0-4-0 (2023-06-09)
~~~~~~~~~~~~~~~~~~~~~~~~

Break
~~~~~

- Removed the spectra_format configuration of SBART


Bugfix
~~~~~~

- Fixing telfit access to its default profile
- Avoid multiple data reloads of S1D data from ESPRESSO
- Fixing mutability issue on the class properties of the instruments
- Properly propagate the user-provided target name  (`#31 <https://github.com/iastro-pt/sBART/issues/31>`_)
- Removing calls to np.bool  (`#33 <https://github.com/iastro-pt/sBART/issues/33>`_)


Feature
~~~~~~~

- Adding easier way of bulk-normalizing stellar spectra
- Allow to transform S1D Frame into S2D
- Updated the interface for the DataClass
- Add fallback for Telfit templates for reference frames for which we can't download GDAS profile
- Allow to select observations from disk files (update to SpectralConditions)
- Loading ESPRESSO binX and binY
- Adding the DLW algorithm as an activity indicator
- Automatic recognition of filetypes


Misc
~~~~

- Standardizing the API interface for the samplers


SBART 0-1-6 (2022-11-02)
~~~~~~~~~~~~~~~~~~~~~~~~

Break
~~~~~

- Fine control of interpolation now made with API calls to a DataClass object 
- NUMBER_WORKERS of Stellar Template and RV routines now only control the number of workers; 


Bugfix
~~~~~~

- Fixing bug with CCF RV uncertainty load from disk 
- Fixing circular import on the Parameters sub-package 
- Fix SA correction if the BJD was not loaded by the instruments 
- Removed bug on Telluric Template creation without water-related keywords  (`#25 <https://github.com/iastro-pt/sBART/issues/25>`_)
- Ensuring that the template and the spectra have the same corrections  (`#27 <https://github.com/iastro-pt/sBART/issues/27>`_)


Feature
~~~~~~~

- Fallback of BJD missing key to MJD 
- Sigma-clipping RVs based on the DRS estimate 
- Plotting transmittance spectrum 
- Allow to reject Frames based on warning KW flags that might exist 
- Provide framework to normalize the stellar spectra (currently RASSINE) 
- Introducing GP interpolation for stellar spectra (Frames and Stellar Templates) 
- Introducing NearestNeighbor interpolation option 


Misc
~~~~

- Adding sanity tests to (RV) model parameters 
- Added detail to worker log from RV_routine 
- Improving logs to avoid repeating multiple times the same information about the Frames user-configs 
- Improving the interface for interpolating stellar spectra 
- Added bump2version to automatically bump version numbers  (`#4 <https://github.com/iastro-pt/sBART/issues/4>`_)
- Adding docs for the disk outputs of the pipeline  (`#29 <https://github.com/iastro-pt/sBART/issues/29>`_)


SBART 0-1-6 (2022-05-19)
------------------------

Features
~~~~~~~~

- Added interface to the external_tools to load RV results from disk (`#23 <https://github.com/iastro-pt/sBART/issues/23>`_)


Bugfixes
~~~~~~~~

- Removed circular import on RV loading function (`#23 <https://github.com/iastro-pt/sBART/issues/23>`_)


SBART 0-1-5 (2022-05-13)
------------------------

Features
~~~~~~~~

- Standardizing the disk names of Stellar and Telluric templates (`#12 <https://github.com/iastro-pt/sBART/issues/12>`_)


Bugfixes
~~~~~~~~

- Not allowing to try load of S1D data of non-ESPRESSO instruments (`#10 <https://github.com/iastro-pt/sBART/issues/10>`_)
- Properly flagging mandatory parameter with no value (`#11 <https://github.com/iastro-pt/sBART/issues/11>`_)
- Fixed bug that prevented previous Stellar templates from being loaded from disk (`#12 <https://github.com/iastro-pt/sBART/issues/12>`_)
- Ensuring that RV routines get a pathlib.Path object even if a str is passed (`#14 <https://github.com/iastro-pt/sBART/issues/14>`_)
- Laplace sampler no longer raises an error when trying to access a (now) non-existing parameter (`#18 <https://github.com/iastro-pt/sBART/issues/18>`_)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Improving quickstart guide to use a pathlib.Path as the storage option (`#14 <https://github.com/iastro-pt/sBART/issues/14>`_)


SBART 0-1-4 (2022-05-03)
------------------------

Bugfixes
~~~~~~~~

- Fixed installation for python3.9 (`#8 <https://github.com/iastro-pt/sBART/issues/8>`_)


SBART 0-1-0 (2022-04-29)
------------------------

Features
~~~~~~~~

- Allowing to run SBART from single function (`#6 <https://github.com/iastro-pt/sBART/issues/6>`_)


Bugfixes
~~~~~~~~

- Fixing missing imports that crept it (`#7 <https://github.com/iastro-pt/sBART/issues/7>`_)


SBART 0-1-0 (2022-04-28)
------------------------

Features
~~~~~~~~

- Added TelFit support (`#1 <https://github.com/iastro-pt/sBART/issues/1>`_)
- Allow to use MAD to flag flux outliers. (`#2 <https://github.com/iastro-pt/sBART/issues/2>`_)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Add your info here (`#3 <https://github.com/iastro-pt/sBART/issues/3>`_)


SBART 0-0-0 (2022-04-27)
------------------------

Features
~~~~~~~~

- First release of the SBART pipeline
