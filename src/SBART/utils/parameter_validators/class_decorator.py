from functools import wraps


def validator(func):  # pylint: disable=no-self-argument
    """
    Used to validate the error flag before running a function.
    Decorator used inside this class

    Returns
    -------

    """

    @wraps(func)
    def on_call(self, *args, **kwargs):
        if not self.is_open:
            raise Exception("The RV routine is not open")
        return func(self, *args, **kwargs)  # pylint: disable=not-callable

    return on_call


def argument_checker(func):  # pylint: disable=no-self-argument
    """
    Used to validate the error flag before running a function.
    Decorator used inside this class


    TODO: understand why this exists and if I am using this... .
    Returns
    -------

    """

    @wraps(func)
    def on_call(self, *args, **kwargs):
        if self._internal_storage:
            return func(self, *args, **self._internals)

        return func(self, *args, **kwargs)  # pylint: disable=not-callable

    return on_call
