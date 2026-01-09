"""Constants for {{ cookiecutter.package_name }}."""

import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_FILE = os.path.join(ROOT_PATH, "config.ini")

DEFAULT_TMP_OUTPUT_PATH = os.path.join(ROOT_PATH, "tmp")

DEFAULT_LOG_PATH = os.path.join(ROOT_PATH, "logs")