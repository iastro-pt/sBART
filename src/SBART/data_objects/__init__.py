"""
Classes that handle input/output data.

"""
from SBART.utils.concurrent_tools.proxyObjects import DataClassManager

from .DataClass import DataClass
from .MetaData import MetaData
from .RV_cube import RV_cube
from .RV_outputs import RV_holder

DataClassManager.register("DataClass", DataClass)

from .Target import Target
