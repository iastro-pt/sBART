import warnings

# import signal
import numpy as np
from astropy import units as u
from astroquery.simbad import Simbad

from SBART.utils.units import meter_second

as_yr = u.arcsec / u.year
mas_yr = u.milliarcsecond / u.year
mas = u.milliarcsecond


def handler(signum, frame):
    print("Timeout!")
    raise Exception


def _de_escape(star):
    if "TOI-" in star:
        return star.replace("TOI-", "TOI ")
    return star


def build_query(star):
    star = _de_escape(star)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    aSimbad = Simbad()
    aSimbad.add_votable_fields("rv_value", "pm", "pmdec", "pmra", "plx", "ra(d)", "dec(d)")

    r = aSimbad.query_object(star)
    if r is None:
        raise ValueError(f"star {star} not found in Simbad")
    else:
        r = r[0]
    query = f"""SELECT TOP 1 * FROM gaiadr2.gaia_source \
    WHERE CONTAINS(
                POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),
                CIRCLE('ICRS',
                    COORD1(EPOCH_PROP_POS({r['RA_d']},{r['DEC_d']},{r['PLX_VALUE']},{r['PMRA']},{r['PMDEC']},{r['RV_VALUE']},2000,2015.5)),
                    COORD2(EPOCH_PROP_POS({r['RA_d']},{r['DEC_d']},{r['PLX_VALUE']},{r['PMRA']},{r['PMDEC']},{r['RV_VALUE']},2000,2015.5)),
                    0.001388888888888889))=1
    """

    return r, query


def run_job(query, timeout=5):
    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(timeout)
    from astroquery.gaia import Gaia  # avoid print at import time

    job = Gaia.launch_job(query)
    return job.get_results()


def mu(pmra, pmdec):
    return np.sqrt(pmra**2 + pmdec**2)


def secular_acceleration(star):
    rsimbad, query = build_query(star)

    r = run_job(query)[0]
    pmra = (r["pmra"] * mas_yr).to(as_yr).value
    pmdec = (r["pmdec"] * mas_yr).to(as_yr).value
    p = (r["parallax"] * mas).to(u.arcsec).value

    if p == 0.0:
        # print(f'Warning: No measured GAIA parallax for {star}. Using Simbad')
        pmra = (rsimbad["PMRA"] * mas_yr).to(as_yr).value
        pmdec = (rsimbad["PMDEC"] * mas_yr).to(as_yr).value
        p = (rsimbad["PLX_VALUE"] * mas).to(u.arcsec).value

    sa = 0.0229 * mu(pmra, pmdec) ** 2 / p  # m/s/yr
    return (
        sa * meter_second
    )  # not really the proper units, but close enough!! The year is taken into account in the RV correction
