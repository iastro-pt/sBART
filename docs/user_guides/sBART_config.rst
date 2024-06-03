=======================================
Configuration of the different modules
=======================================

The vast majority of SBART values can be easily configured by the user. However, the majority
of them also takes sensible default values for a general application.


Each SBART object, as described in the :ref:`Public interface <SbartAPI>`, can define a list of "User Parameters",
where we present:

- Parameter name - the key in the config dictionary
- Mandatory: a boolean value to represent if it is mandatory or not.
- Default Value - If it is not mandatory and not provided by the user, this will be the value
- Valid Values - Conditions that the user-given parameter **must** comply with
- Comment - description of the goal of the parameter


On top of that, each SBART object also has a "Base", which can introduce extra "User Parameters" that are also
available to be configured.

.. note::

    Lets take a look at a specific case, the one from :py:class:`~SBART.Instruments.ESPRESSO`.
    This class provides two "User parameters. However, its "Base" is the  :py:class:`~SBART.Base_Models.Frame`
    class, which introduces 4 other "User parameters".

    Thus, the user can change the following fields of a given ESPRESSO observation:

    - apply_FluxCorr
    - Telluric_Corrected
    - bypass_QualCheck
    - reject_order_percentage
    - minimum_order_SNR
    - spectra_format

Each different piece of SBART defines a set of parameters, which the user can override by providing
a python dictionary, where the keys are the name of the parameter, and the values are the desired
value. The user-provided values are then passed by a validation stage, raising an Exception if the
user-provided value does not comply with the conditions that the parameter imposes.

.. note::

    The SBART objects normally provide a *user_configs* argument in their constructor that will accept
    the configuration dictionary. If this is not the case, explicit information will be given.

