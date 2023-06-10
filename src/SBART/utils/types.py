from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from astropy.units import Quantity

RV_measurement = Quantity

data_vector = Iterable[Quantity]

unitless_data_vector = Union[List[float], np.ndarray]

unitless_data_matrix = np.ndarray


UI_PATH = Union[str, Path]
UI_DICT = Dict[str, Any]
