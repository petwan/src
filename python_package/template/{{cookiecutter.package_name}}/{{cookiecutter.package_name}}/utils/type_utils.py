"""Type utility definitions for {{ cookiecutter.package_name }}."""

from typing import Union, TypeVar, Optional


T = TypeVar('T')
Number = Union[int, float]
Maybe = Optional[T]