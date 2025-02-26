# pylint=disable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Tuple

import pytest

from SBART.utils import UserConfigs as parameter_validators
from SBART.utils.custom_exceptions import InternalError, InvalidConfiguration
from SBART.utils.choices import DISK_SAVE_MODE


@pytest.mark.parametrize(
    "test_input,mode,expectation",
    [
        ([1, 2, 3], "all", nullcontext(1)),
        ((1, 2, 3), "all", nullcontext(1)),
        ([1, 2], "all", pytest.raises(InvalidConfiguration)),
        ([55], "either", pytest.raises(InvalidConfiguration)),
    ],
)
def test_IterableMustHave(test_input: Iterable, mode: str, expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = parameter_validators.IterableMustHave(available_options=(1, 2, 3), mode=mode)
        validator.evaluate("foo", test_input)


@pytest.mark.parametrize(
    "test_input,expectation",
    [
        ([1, 2, 3], pytest.raises(InvalidConfiguration)),
        (1, nullcontext(1)),
        (None, nullcontext(1)),
        ("text", nullcontext(1)),
    ],
)
def test_ValueFromIterable(test_input: Any, expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = parameter_validators.ValueFromIterable(available_options=(1, None, "text"))
        validator.evaluate("foo", test_input)


@pytest.mark.parametrize(
    "test_input,expectation",
    [
        ("BASIC", pytest.raises(InvalidConfiguration)),
        (DISK_SAVE_MODE.BASIC, nullcontext(1)),
    ],
)
def test_ValueFromEnum(test_input: Any, expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = parameter_validators.ValueFromIterable(available_options=DISK_SAVE_MODE)
        validator.evaluate("foo", test_input)


@pytest.mark.parametrize(
    "test_input,include_edges,expectation",
    [
        (0, False, pytest.raises(InvalidConfiguration)),
        (5, False, pytest.raises(InvalidConfiguration)),
        (6, False, pytest.raises(InvalidConfiguration)),
        (6, True, pytest.raises(InvalidConfiguration)),
        (5, True, nullcontext()),
        (0, True, nullcontext()),
        (2, True, nullcontext()),
    ],
)
def test_ValueInInterval(test_input: int, include_edges: bool, expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = parameter_validators.ValueInInterval(interval=(0, 5), include_edges=include_edges)
        validator.evaluate("foo", test_input)


@pytest.mark.parametrize(
    "test_input,dtypes,expectation",
    [
        (0.0, (int,), pytest.raises(InvalidConfiguration)),
        (0, (int,), nullcontext()),
    ],
)
def test_ValueFromDtype(test_input: Any, dtypes: Tuple[type, ...], expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = parameter_validators.ValueFromDtype(dtypes)
        validator.evaluate("foo", test_input)


@pytest.mark.parametrize(
    "test_input,include_edges,expectation",
    [
        (0, False, pytest.raises(InvalidConfiguration)),
        (5, False, pytest.raises(InvalidConfiguration)),
        (6, False, pytest.raises(InvalidConfiguration)),
        (4.5, True, pytest.raises(InvalidConfiguration)),
        (1, True, nullcontext()),
        (2, True, nullcontext()),
    ],
)
def test_sum_of_conds(test_input: Any, include_edges: bool, expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = (
            # parameter_validators.ValueInInterval(interval=(0, 5), include_edges=include_edges)
            parameter_validators.ValueFromIterable((1, 2, 3, 4.5)) + parameter_validators.ValueFromDtype((int,))
        )
        validator.evaluate("foo", test_input)


@pytest.mark.parametrize(
    "test_input,expectation",
    [
        (Path(__file__), nullcontext()),
    ],
)
def test_PathValue(test_input: Path, expectation: AbstractContextManager) -> None:
    with expectation as e:
        validator = parameter_validators.PathValue
        validator.evaluate("foo", test_input)
