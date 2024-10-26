import SBART
from SBART import *
from pathlib import Path

docs_path = Path(__file__).parent / "Userconfigs"
import numpy as np

print(
    dir(SBART),
)
print(SBART.Instruments)
print(dir(SBART.utils.cython_codes.matmul))
