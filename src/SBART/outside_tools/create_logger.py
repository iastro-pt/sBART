"""Setup the logger, handling terminal and disk output."""

import sys

from loguru import logger

# Keep a reference to ASTRA's sink IDs so we can remove them later if needed
_SBART_SINKS = []
_SBART_INITIALIZED = False

# Default logger (can be reconfigured later)
sbart_logger = logger.bind(module="SBART")
logger.disable("SBART")


def setup_sbart_logger(
    RV_method: str, log_path=None, level="INFO", log_to_terminal=True, write_to_file=True, append_to_file=True
):
    """Configure SBART's logger dynamically.

    Parameters
    ----------
    log_path : str or Path or None
        Path to the log file. If None, file logging is skipped.
    level : str
        Minimum level to log ("DEBUG", "INFO", "WARNING", etc.)
    console : bool
        Whether to also log to stdout.
    """
    global _SBART_INITIALIZED, _SBART_SINKS

    if _SBART_INITIALIZED:
        # Clean up previous sinks (safe for reconfiguration)
        for sink_id in _SBART_SINKS:
            logger.remove(sink_id)
        _SBART_SINKS.clear()

    # Define filter so only ASTRA messages appear
    def sbart_filter(record):
        return record["extra"].get("module") == "SBART"

    sbart_logger = logger.bind(module="SBART")
    logger.enable("SBART")
    # Optional console logging

    if log_to_terminal:
        _SBART_SINKS.append(
            sbart_logger.add(
                sys.stdout,
                level=level,
                filter=sbart_filter,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> {name} <level>{message}</level>",
            )
        )

    # Optional file logging
    if log_path is not None and write_to_file:
        _SBART_SINKS.append(
            sbart_logger.add(
                (log_path / f"{RV_method}.log").as_posix(),
                level=level,
                filter=sbart_filter,
                format="{time:YYYY-MM-DD HH:mm:ss} | {name} {level} | {message}",
                enqueue=True,
                mode="a" if append_to_file else "w",
            )
        )

    _SBART_INITIALIZED = True
    return sbart_logger
