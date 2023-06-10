import os
from datetime import datetime

from dateutil.relativedelta import relativedelta  # $ pip install python-dateutil


def file_older_than(file: str, days: int) -> bool:
    """Check if `file` has been modified more than `days` days ago"""
    threshold_date = datetime.now() - relativedelta(days=days)
    file_time = datetime.fromtimestamp(os.path.getmtime(file))
    return file_time < threshold_date
