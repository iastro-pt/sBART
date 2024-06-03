def check_variation_inside_interval(
    previous: float, current: float, percentage_threshold: float
) -> bool:
    """
    Check if a value changed less than a given percentage between previous and current measurement
    Parameters
    ----------
    previous: OLd value
    current: New value
    percentage_threshold: Max variation allowed

    Returns
    -------

    """
    return abs((previous - current) / current) < percentage_threshold
