================
Installation
================

SBART can be pip-installed:

.. code-block:: console

    pip install SBART


To use `Telfit <https://github.com/kgullikson88/Telluric-Fitter>`_. it must be installed independently:

.. code-block:: console

    pip install Telfit

.. note::
    Ensure that you have the BLAS + lapak libraries installed, as they will be needed to compile the *.cython* files. If they are not available, the installation will fail


To install from github (for recent, but not pypi-published changes), you should only need to:

1) /git clone/ the repository
2) *cd* into it
3) pip install .