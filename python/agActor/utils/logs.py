import logging


def log_message(logger: logging.Logger | None, message: str, level: str | int = "INFO"):
    """Log a message at the specified log level if logger is provided.

    Args:
        logger: Logger object to use for logging. If None, no logging occurs.
        message: The message string to log.
        level: The logging level as string or int. Defaults to "INFO".
    """
    if logger is not None:
        if isinstance(level, str):
            try:
                level = getattr(logging, level.upper())
            except AttributeError:
                raise ValueError(f"Invalid logging level: {level}")

        logger.log(level, message)
