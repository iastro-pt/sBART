==========================
Goals of this tutorial
==========================

1. See how we can normalize the stellar spectra

-----------------------------
Handling individual Frames
-----------------------------

The low level approach to the normalization of stellar spectra is to manually load the Frames and call the relevant functions (S1D spectra also works)

To trigger the normalization we must now pass extra information in the config dictionary:

.. code-block:: python

    frame_information = {"NORMALIZE_SPECTRA": True,
                         "NORMALIZATION_MODE": "RASSINE",
                     }

The "NORMALIZATION_MODE" will be used to select the method that will be used to normalize the stellar spectra and it can support the algorithms described in :py:class:`~SBART.Components.Spectral_Normalization`. Please refer to the page of the individual algorithms to understand the inputs that they need.

.. note:: Keep in mind that the current implementation of the spectral normalization will break all RV extraction algorithms. Do **not** use this functionality before trying to extract RVs.


---------------------
High level approach
---------------------

If we don't want to directly deal with the Frames we can call the normalization interface through the DataClass:


After this point we can either launch the normalization for **all** frames:

.. code-block:: python

    data_object.normalize_all()


Or handle the individual subInstruments:

.. code-block:: python

    data_object.normalize_all_from_subInst(data_object.get_subInstruments_with_valid_frames()[0])