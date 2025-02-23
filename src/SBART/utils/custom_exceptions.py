from loguru import logger


class Error(Exception):
    """Base class for exceptions in this module."""



class FrameError(Error):
    """Raised when a INVALID S2D file is opened!"""



class AlreadyLoaded(Error):
    """Raised when we try to load something that was already loaded!"""



class BadOrderError(Error):
    """Raised when we access a bad order"""



class NoConvergenceError(Error):
    """Raised when we access data from a fit that did not converge"""



class NoComputedRVsError(Error):
    """Used when trying to access to a RV cue that was not computed

    Parameters
    ----------
    Error : [type]
        [description]

    """



class DeadWorkerError(Error):
    """Used when a worker finds a problem"""



class NoDataError(Error):
    """Used when all loaded frames are blocked/invalid"""



class InvalidConfiguration(Error):
    """Used when a given configuration falls outside the accepted values"""



class TemplateNotExistsError(Error):
    """Used whenever we try to access a template that does not exist"""


class BadTemplateError(Error):
    """Used whenever we try to access a template that failed its computation"""


class FailedStorage(Error):
    """Used whenever we try to access a template that failed its computation"""


class NoDataError(Error):
    """Used whenever we are left with no data to proccess"""


class InternalError(Error):
    """Used whenever we have an error with the non-user part of the code"""


class MissingRootPath(Error):
    """Used whenever we have an error with the non-user part of the code"""


class StopComputationError(Error):
    """Used whenever we have an error with the non-user part of the code"""


def ensure_invalid_template(func):
    def inner1(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except Exception:
            self.mark_as_invalid()
            logger.opt(exception=True).critical("Template creation failed")

    return inner1
