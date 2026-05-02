import dataclasses
import functools
import typing
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import overload
from typing import ParamSpec
from typing import Protocol
from typing import runtime_checkable
from typing import TypeGuard
from typing import TypeVar

import chex
import jax.experimental.checkify
import torch

K = TypeVar("K")
V = TypeVar("V")
Aux = TypeVar("Aux")
In = TypeVar("In")
P = ParamSpec("P")
Out_cov_co = TypeVar("Out_cov_co", covariant=True)


type NestedDict[K, V] = dict[K, V | NestedDict[K, V]]
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]

type PyTree[T] = T | tuple[PyTree[T], ...] | list[PyTree[T]] | dict[Any, PyTree[T]]

type Scalar = float | int | bool
type JaxPyTree = Scalar | jax.Array | tuple[JaxPyTree, ...] | list[JaxPyTree] | Mapping[Any, JaxPyTree]
type TorchPyTree = Scalar | torch.Tensor | tuple[TorchPyTree, ...] | list[TorchPyTree] | Mapping[Any, TorchPyTree]
Params = TypeVar("Params", bound=JaxPyTree)


T = TypeVar("T", jax.Array, torch.Tensor)


@runtime_checkable
class Module(Protocol[P, Out_cov_co]):
    """Protocol for a torch.nn.Module that gives better type hints for the `__call__` method."""

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Out_cov_co:
        raise NotImplementedError

    if typing.TYPE_CHECKING:
        # note: Only define this for typing purposes so that we don't actually override anything.
        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Out_cov_co: ...

        modules = torch.nn.Module.modules
        named_modules = torch.nn.Module.named_modules
        state_dict = torch.nn.Module.state_dict
        zero_grad = torch.nn.Module.zero_grad
        parameters = torch.nn.Module.parameters
        named_parameters = torch.nn.Module.named_parameters
        cuda = torch.nn.Module.cuda
        cpu = torch.nn.Module.cpu
        # note: the overloads on nn.Module.to cause a bug with missing `self`.
        # This shouldn't be a problem.
        to = torch.nn.Module().to


# NOTE: Not using a `runtime_checkable` version of the `Dataclass` protocol here, because it
# doesn't work correctly in the case of `isinstance(SomeDataclassType, Dataclass)`, which returns
# `True` when it should be `False` (since it's a dataclass type, not a dataclass instance), and the
# runtime_checkable decorator doesn't check the type of the attribute (ClassVar vs instance
# attribute).


class _DataclassMeta(type):
    def __subclasscheck__(self, subclass: type) -> bool:
        return dataclasses.is_dataclass(subclass) and not dataclasses.is_dataclass(type(subclass))

    def __instancecheck__(self, instance: Any) -> bool:
        return dataclasses.is_dataclass(instance) and dataclasses.is_dataclass(type(instance))


class Dataclass(metaclass=_DataclassMeta):
    """A class which is used to check if a given object is a dataclass.

    This plays nicely with @functools.singledispatch, allowing us to register functions to be used
    for dataclass inputs.
    """


class DataclassInstance(Protocol):
    # Copy of the type stub from dataclasses.
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


DataclassType = TypeVar("DataclassType", bound=DataclassInstance)


def is_sequence_of[V](object: Any, item_type: type[V] | tuple[type[V], ...]) -> TypeGuard[Sequence[V]]:
    """Used to check (and tell the type checker) that `object` is a sequence of items of type
    `V`."""
    try:
        return all(isinstance(value, item_type) for value in object)
    except TypeError:
        return False


def is_list_of[V](object: Any, item_type: type[V] | tuple[type[V], ...]) -> TypeGuard[list[V]]:
    """Used to check (and tell the type checker) that `object` is a list of items of this type."""
    return isinstance(object, list) and is_sequence_of(object, item_type)


def jit[**P, Out](fn: Callable[P, Out]) -> Callable[P, Out]:
    """Small type hint fix for jax's `jit` (preserves the signature of the callable)."""
    return jax.jit(fn)  # type: ignore


# argnums = 0
@overload
def value_and_grad[In, *Ts, Out](
    fn: Callable[[In, *Ts], Out],
    argnums: Literal[0] = 0,
    has_aux: bool = ...,
) -> Callable[[In, *Ts], tuple[Out, In]]: ...


@overload
def value_and_grad[In, In2, *Ts, Out](
    fn: Callable[[In, In2, *Ts], Out],
    argnums: tuple[Literal[0], Literal[1]],
    has_aux: bool = ...,
) -> Callable[[In, *Ts], tuple[Out, tuple[In, In2]]]: ...


@overload
def value_and_grad[In, In2, In3, *Ts, Out](
    fn: Callable[[In, In2, In3, *Ts], Out],
    argnums: tuple[Literal[0], Literal[1], Literal[2]],
    has_aux: bool = ...,
) -> Callable[[In, *Ts], tuple[Out, tuple[In, In2, In3]]]: ...


@overload
def value_and_grad[In, *Ts, Out](
    fn: Callable[[In, *Ts], Out],
    argnums: tuple[Literal[0], *tuple[int, ...]],
    has_aux: bool = ...,
) -> Callable[[In, *Ts], tuple[Out, tuple[In, *Ts]]]: ...


@overload
def value_and_grad[*Ts, Out](
    fn: Callable[[*Ts], Out],
    argnums: Sequence[int],
    has_aux: bool = ...,
) -> Callable[[*Ts], tuple[*Ts]]: ...


def value_and_grad[Out](  # type: ignore
    fn: Callable[..., Out],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
):
    """Small type hint fix for jax's `value_and_grad` (preserves the signature of the callable)."""
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)


def chexify[**P, Out](
    fn: Callable[P, Out],
    async_check: bool = True,
    errors: frozenset[jax.experimental.checkify.ErrorCategory] = chex.ChexifyChecks.user,
) -> Callable[P, Out]:
    """Fix `chex.chexify` so it preserves the function's signature."""
    return functools.wraps(fn)(chex.chexify(fn, async_check=async_check, errors=errors))  # type: ignore
