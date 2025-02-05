import numpy as np 
from SBART.utils.json_ready_converter import json_ready_converter
import json 


def test_converter():
    test = {"foo": np.zeros(3),
            "bar": np.zeros(1)[0]
            }
    
    for key, val in test.items():
        print(key, type(val), isinstance(val, (np.floating, np.integer)))
    
    json.dumps(json_ready_converter(test))