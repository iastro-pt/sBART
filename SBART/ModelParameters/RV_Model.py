from typing import List

from SBART.ModelParameters.Model import Model
from SBART.ModelParameters.Parameter import ModelComponent
from SBART.utils import custom_exceptions


class RV_Model(Model):
    def __init__(self, params_of_model: List[ModelComponent]):
        super().__init__(params_of_model)

        if params_of_model[0].param_name != "RV":
            raise custom_exceptions.InvalidConfiguration(
                "The first component of the model should be the RV"
            )

    def get_RV_bounds(self, frameID):
        return self.get_component_by_name("RV").get_bounds(frameID=frameID)
