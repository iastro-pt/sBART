from SBART.Instruments import ESPRESSO


def test_data_transform():
    frame = ESPRESSO(file_path="/home/amiguel/phd/spectra_collection/ESPRESSO/TauCeti/r.ESPRE.2022-07-06T09:22:31.285_S1D_A.fits")

    new_frame = frame.copy_into_S2D()

    new_frame.close_arrays()

    assert frame.sub_instrument == new_frame.sub_instrument
    assert new_frame.wavelengths is not None
    assert new_frame.spectra is not None
    assert new_frame.uncertainties is not None

    assert new_frame.wavelengths.shape == (170, 9111)
    assert new_frame.spectral_format == "S2D"

    assert all(new_frame.wavelengths[160] == 0)

    new_frame.build_mask()
