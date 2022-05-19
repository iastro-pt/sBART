=============
Release Notes
=============

.. towncrier release notes start

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
