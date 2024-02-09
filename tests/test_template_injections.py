from pathlib import Path

import numpy as np

curr_folder = Path(__file__).parent

from typing import List, Any, Dict, Union, Optional

from SBART.Instruments import ESPRESSO
from SBART.template_creation import SumStellar
from SBART.data_objects import DataClassManager, DataClass
from template_creation.StellarModel import StellarModel

possible_frames = Path("/home/amiguel/phd/spectra_collection/ESPRESSO/HD10700_2series1night")


def _construct_template(filelist: List[Path], storage_path: Optional[str]) -> Union[SumStellar, Dict[str, Any]]:
    frame_load_configs = {"apply_FluxCorr": True,
        }

    manager = DataClassManager()
    manager.start()

    data: DataClass = manager.DataClass(
        filelist,
        storage_path=Path(storage_path),
        instrument=ESPRESSO,
        instrument_options=frame_load_configs,
        )

    subInsts = data.get_subInstruments_with_valid_frames()
    if len(subInsts) != 1:
        raise Exception

    ModelStell = StellarModel(
                root_folder_path=Path(storage_path)
        )

    ModelStell.Generate_Model(dataClass=data, template_configs={})
    if storage_path is not None:
        ModelStell.store_templates_to_disk()

    return ModelStell.request_data(subInsts[0]), frame_load_configs


def test_template_injections():
    available_S2D = list(possible_frames.glob("**/*S2D_A.fits"))

    original_template_list = available_S2D[:6]
    full_template_list = available_S2D[:7]

    first_template, frame_configs = _construct_template(original_template_list,
                                                        storage_path=curr_folder/"to_inject")

    w,c,e,m = first_template.get_data_from_spectral_order(5)

    frame_to_inject = ESPRESSO(
        file_path=full_template_list[-1],
        user_configs=frame_configs,
        )

    first_template.add_new_frame_to_template(frame_to_inject,
                                             )

    complete_template, _ = _construct_template(full_template_list,
                                               storage_path=curr_folder/"complete")

    # TODO: add comparison of the complete and the injected template

    for order in range(frame_to_inject.N_orders):

        wave_C, flux_C, err_C, mask_C = complete_template.get_data_from_spectral_order(order=order,
                                                                               include_invalid=False
                                                                               )

        wave_i, flux_i, err_i, mask_i = first_template.get_data_from_spectral_order(
            order=order,
            include_invalid=False
            )
        # print(order, np.allclose(flux_C, flux_i), np.allclose(err_C, err_i), np.all(mask_C==mask_i))

        # if order == 5:
        #     INDS = np.where(mask_i != mask_C)
        #     print(np.where(mask_i != mask_C), mask_i[INDS], mask_C[INDS], m[INDS])
        assert np.allclose(flux_C, flux_i)
        assert np.allclose(err_C, err_i)
        assert np.all(mask_C == mask_i)




if __name__ == "__main__":
    test_template_injections()
