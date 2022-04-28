import asyncio
import os
import sys
import warnings

from loguru import logger

# warnings.simplefilter('error', UserWarning)


def setup_SBART_logger(storage_path: str, RV_method: str, instrument,
                       log_to_terminal: bool = True,
                       terminal_log_level='DEBUG', write_to_file=True) -> None:
    """Call from outside the module to setup the logger and necessary folders + structure

    FIXME: This will create empty log files for all sub_instruments of the instrument, even if it has no data!
    Parameters
    ----------
    path : str
        Path to the folder in which the logs will be stored
    RV_method : str
        [description]
    instrument: pyROAST.instruments
        Instrument that will be used
    """

    logger.enable("SBART")
    logger.complete()

    available_blocks = instrument.sub_instruments.keys()

    logger.remove()

    logger.level("DEBUG", color="<fg #d0d3d4>")
    logger.level("INFO", color="<fg #28b463>")
    logger.level("WARNING", color="<fg #f1c40f>")
    logger.level("CRITICAL", color="<fg #e74c3c>")

    fmt = "{time:YYYY-MM-DDTHH:mm:ss} - {name} - {level} - {message}"
    if log_to_terminal:
        logger.add(
            sys.stdout,
            level=terminal_log_level,
            colorize=True,
            format="{time:YYYY-MM-DDTHH:mm:ss} - <c>{name}</> - <level>{level}</> - {message}",
        )
        # but we do want to see the values, so I don't really care about this!
        logger.add(sys.stderr, level="ERROR", format=fmt)

    if not write_to_file:
        logger.warning("Not storing logs to disk")
        return
    logger.add(
        os.path.join(storage_path, f"{RV_method}.log"),
        level="DEBUG",
        format=fmt,
        enqueue=True,
        filter="SBART",
        backtrace=False,
        diagnose=True,
    )

    storage_method_folder = os.path.join(storage_path, RV_method)
    try:
        os.mkdir(storage_method_folder)
    except OSError:
        pass

    for key in available_blocks:
        logger.add(
            os.path.join(storage_method_folder, f"{key}.log"),
            filter=lambda record: record["extra"].get("name") == key,
        )

    # always preserve the logs from the creation of the templates
    # These two are also propagated in the main logger!
    logger.add(
        os.path.join(storage_method_folder, "StellarTemplate.log"),
        filter=lambda record: record["extra"].get("name") == "StellarTemplate",
    )
    logger.add(
        os.path.join(storage_method_folder, "TelluricTemplate.log"),
        filter=lambda record: record["extra"].get("name") == "TelluricTemplate",
    )

    logger.info(storage_path)


if __name__ == "__main__":
    create_logger("/home/amiguel/seminar/teste_tellurics")
    warnings.warn("warniung ya")
    template_logger = logger.bind(name="StellarTemplate")
    template_logger.info("yo from the stellar template")
    raise Exception("alsjhdkkjahsd")
