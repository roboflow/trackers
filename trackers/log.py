import logging
import os
import sys
from typing import Any, Dict, Final, Literal, Optional

_LOG_LEVELS: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_LOG_FILENAME: Final[str] = os.environ.get("TRACKERS_LOG_FILENAME", "trackers.log")
_LOG_LEVEL_NAME: Final[str] = os.environ.get("TRACKERS_LOG_LEVEL", "ERROR").upper()
_LOG_OUTPUT_TYPE: Final[str] = os.environ.get("TRACKERS_LOG_OUTPUT", "stderr").lower()
_LOG_LEVEL: Final[int] = _LOG_LEVELS.get(_LOG_LEVEL_NAME, logging.ERROR)
_LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class LogFormatter(logging.Formatter):
    """
    Custom log formatter that adds ANSI color codes to log messages based on
    the log level for terminal output. Does not add color codes if the output
    is redirected to a file. This formatter is designed to work with Python 3.10+.
    It uses ANSI escape sequences to colorize log messages for better visibility
    in terminal environments.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,
        *,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        if sys.version_info >= (3, 10):
            super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        else:
            super().__init__(fmt, datefmt, style, validate)

        self._RESET: Final[str] = "\x1b[0m"

        self._COLOURS: Final[dict[int, str]] = {
            logging.DEBUG: "\x1b[38;21m",
            logging.INFO: "\x1b[34;1m",
            logging.WARNING: "\x1b[33;1m",
            logging.ERROR: "\x1b[31;1m",
            logging.CRITICAL: "\x1b[35;1m",
        }

        self._BASE_FORMAT: Final[str] = "%(asctime)s - %(name)s - "
        self._LEVEL_MSG_FORMAT: Final[str] = "%(levelname)s: %(message)s"

        self._FORMATS: dict[int, str] = {
            level: color + self._BASE_FORMAT + self._LEVEL_MSG_FORMAT + self._RESET
            for level, color in self._COLOURS.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record with color based on the log level.
        Args:
            record (logging.LogRecord): The log record to format.
        Returns:
            str: The formatted log message with color.
        """

        log_fmt = self._FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


if _LOG_OUTPUT_TYPE == "file":
    logging.basicConfig(
        level=_LOG_LEVEL,
        format=_LOG_FORMAT,
        filename=_LOG_FILENAME,
        filemode="a",
    )
else:
    root_logger = logging.getLogger()
    root_logger.setLevel(_LOG_LEVEL)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(LogFormatter())
    root_logger.addHandler(handler)


def get_logger(name: Optional[str]) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    Args:
        name (str): The name for the logger, typically __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
