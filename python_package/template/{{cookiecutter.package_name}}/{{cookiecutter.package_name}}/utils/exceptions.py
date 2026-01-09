"""Custom exceptions for {{ cookiecutter.package_name }}."""


class {{ cookiecutter.package_name.replace('_', ' ').title().replace(' ', '') }}Error(Exception):
    """Base exception class for {{ cookiecutter.package_name }} errors."""
    pass