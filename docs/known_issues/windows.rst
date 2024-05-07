=======================================
Issues that appear for Windows users
=======================================

RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
-------------------------------------------------------------------------------------------------------------------------------------

If you are running sBART in windows installation and the following error appears when starting the DataClass
object:

.. code-block:: console

    RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

There is a (for now) tentative solution: Place the entire script that launches sBART into the following block

.. code-block:: python

    if __name__ == "__main__"
        data = manager.start()
