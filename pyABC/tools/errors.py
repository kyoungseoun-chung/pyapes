#!/usr/bin/env python3
from typing import Optional


class MaximunIterationReachedWarning(RuntimeWarning):
    """Warning when maximum iteration is reached. Used for solvers."""

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)


class SolutionDoesNotConverged(Exception):
    """Warning when maximum iteration is reached. Used for solvers."""

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)


class WrongInputError(Exception):
    """Exception raised when wrong input is given."""

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)


class WrongInputWarning(RuntimeWarning):
    """Exception raised when wrong input is given."""

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)


class SizeDoesNotMatchError(Exception):
    """Exception raised when wrong input is given."""

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)


class ValueDoesNotMatchError(Exception):
    """Exception raised when wrong input is given."""

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)


class FeatureNotImplementedError(Exception):
    """Exception raised when user try to access
    the feature not implemented.
    """

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg)
