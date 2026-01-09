"""Utilities for {{ cookiecutter.package_name }}."""

from .exceptions import {{ cookiecutter.package_name.replace('_', ' ').title().replace(' ', '') }}Error
from .logging_config import setup_logging, get_logger
from .type_utils import Number, Maybe
from .constants import (
    ROOT_PATH,
    DEFAULT_CONFIG_FILE,
    DEFAULT_TMP_OUTPUT_PATH,
    DEFAULT_LOG_PATH
)


__all__ = [
    "{{ cookiecutter.package_name.replace('_', ' ').title().replace(' ', '') }}Error",
    "setup_logging",
    "get_logger",
    "Number",
    "Maybe",
    "ROOT_PATH",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_TMP_OUTPUT_PATH",
    "DEFAULT_LOG_PATH"
]