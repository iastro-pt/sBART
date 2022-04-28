================
Installation
================

SBART can be pip-installed:

.. code-block:: console

    pip install SBART


To use `Telfit <https://github.com/kgullikson88/Telluric-Fitter>`_. it must be installed independently:

.. code-block:: console

    pip install Telfit


Common Issues
================


psycopg2 install fails
---------------------------

If Telfit and/or pysynphot fails with the following error:

.. code-block:: console

    ./psycopg/psycopg.h:36:10: fatal error: libpq-fe.h: No such file or directory


Then, try running the following command and attempting to re-install SBART:

.. code-block:: console

    sudo apt-get install --reinstall libpq-dev
