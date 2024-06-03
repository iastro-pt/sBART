from multiprocessing import Queue
from typing import Tuple

from SBART.utils.status_codes import WORKER_ERROR


def evaluate_shutdown(queue: Queue) -> Tuple[int, int]:
    """Evaluate the shutdown messages sent by the workers, counting the number of good and bad shutdowns

    Parameters
    ----------
    queue : Queue
        Queue in which the workers place their

    Returns
    -------
    Tuple[int, int]
        Goog, bad shutdowns
    """
    good_shutdown = 0
    bad_shutdown = 0
    while not queue.empty():
        pkg = queue.get()
        if pkg == WORKER_ERROR:
            bad_shutdown += 1
        else:
            good_shutdown += 1
    return good_shutdown, bad_shutdown
