================================
Common Issues to all platforms
================================

psycopg2 install fails
------------------------

If Telfit and/or pysynphot fails with the following error:

.. code-block:: console

    ./psycopg/psycopg.h:36:10: fatal error: libpq-fe.h: No such file or directory


Then, try running the following command and attempting to re-install SBART:

.. code-block:: console

    sudo apt-get install --reinstall libpq-dev
