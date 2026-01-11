from __future__ import annotations

import collections.abc
import dataclasses
import functools
import logging
from logging import getLogger as get_logger
from typing import Any, Callable, overload

import jax
import jax.core
import jax.errors
import torch
import torch.func
import torch.utils._pytree
from jax.dlpack import from_dlpack as jax_from_dlpack  # type: ignore

from .types import (
    Dataclass,
    DataclassType,
    K,
    NestedDict,
    NestedMapping,
)
from .utils import (
    log_once,
)

logger = get_logger(__name__)


@overload
def torch_to_jax(value: torch.Tensor, /) -> jax.Array: ...


@overload
def torch_to_jax(value: torch.device, /) -> jax.Device: ...


@overload
def torch_to_jax(value: tuple[torch.Tensor, ...], /) -> tuple[jax.Array, ...]: ...


@overload
def torch_to_jax(value: list[torch.Tensor], /) -> list[jax.Array]: ...


@overload
def torch_to_jax(value: NestedDict[K, torch.Tensor], /) -> NestedDict[K, jax.Array]: ...


@overload
def torch_to_jax(value: Any, /) -> Any: ...


def torch_to_jax(value: Any, /) -> Any:
    """Converts PyTorch tensors to JAX arrays.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: a torch tensor

    Returns:
      a JAX array
    """
    log_once(
        logger,
        message=f"No registered handler for values of type {type(value)}, returning it as-is.",
        level=logging.DEBUG,
    )
    return value


torch_to_jax = functools.singledispatch(torch_to_jax)  # type: ignore


@torch_to_jax.register(type(None))
@torch_to_jax.register(int)
@torch_to_jax.register(float)
@torch_to_jax.register(str)
@torch_to_jax.register(bool)
@torch_to_jax.register(bytes)
def _passthrough_primitive(v: Any) -> Any:
    """Pass through primitive types without conversion."""
    return v


def _direct_conversion(v: torch.Tensor) -> jax.Array:
    return jax_from_dlpack(v, copy=False)


def _to_from_dlpack(v: torch.Tensor) -> jax.Array:
    # Use the newer DLPack API - PyTorch tensors can be passed directly to jax.dlpack.from_dlpack
    # via the __dlpack__ protocol, without needing torch.utils.dlpack.to_dlpack
    return jax_from_dlpack(v, copy=False)


def torch_to_jax_tensor(value: torch.Tensor) -> jax.Array:
    """Converts a PyTorch Tensor into a jax.Array.

    NOTE: torch.float64 tensors may be implicitly converted to jax.float32 tensors.
    """
    value = value.detach()

    # Ensure tensor is contiguous for DLPack compatibility
    if not value.is_contiguous():
        log_once(
            logger,
            message=(
                f"Tensor of shape {tuple(value.shape)} is not contiguous. "
                f"Making a contiguous copy for DLPack conversion."
            ),
            level=logging.DEBUG,
        )
        value = value.contiguous()

    if value.device.type == "cpu":
        try:
            return _to_from_dlpack(value)
        except jax.errors.JaxRuntimeError as err:
            log_once(
                logger,
                message=(
                    f"Unable to convert CPU tensor of shape {tuple(value.shape)} "
                    f"to jax.Array via DLPack: '{err}'"
                ),
                level=logging.WARNING,
            )
            # Fallback: try direct conversion
            return _direct_conversion(value)

    try:
        # Try using the "new" way to convert using from_dlpack directly
        return jax_from_dlpack(
            value, device=torch_to_jax_device(value.device), copy=None
        )
    except AssertionError as err:
        if not err.args[0].startswith("Unexpected XLA layout override"):
            raise
        # Some "AssertionError: Unexpected XLA layout override"
        # Try using the DLPack protocol directly without explicit device
        try:
            return jax_from_dlpack(value, copy=False)
        except jax.errors.JaxRuntimeError as err:
            log_once(
                logger,
                message=(
                    f"Unable to convert GPU tensor of shape {tuple(value.shape)} "
                    f"to jax.Array via DLPack: '{err}'"
                ),
                level=logging.WARNING,
            )
            return _direct_conversion(value)
    except jax.errors.JaxRuntimeError as err:
        log_once(
            logger,
            message=(
                f"Unable to convert GPU tensor of shape {tuple(value.shape)} "
                f"to jax.Array via DLPack: '{err}'"
            ),
            level=logging.WARNING,
        )
        return _direct_conversion(value)


# Register it like this so the type hints are preserved on the functions (which are also called
# directly in some places).
torch_to_jax.register(torch.Tensor, torch_to_jax_tensor)


@torch_to_jax.register(tuple)
def torch_to_jax_tuple(value: tuple) -> tuple:
    return type(value)(*[torch_to_jax(v) for v in value])  # type: ignore


@torch_to_jax.register(list)
def torch_to_jax_list(value: list) -> list:
    return list(torch_to_jax(v) for v in value)


@torch_to_jax.register(collections.abc.Mapping)
def torch_to_jax_dict(
    value: NestedMapping[K, torch.Tensor],
) -> NestedMapping[K, jax.Array]:
    """Converts a dict of PyTorch tensors into a dict of jax.Arrays."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})  # type: ignore


@torch_to_jax.register(Dataclass)
def torch_to_jax_dataclass(value: DataclassType) -> DataclassType:
    """Converts any torch Tensors in the dataclass fields to jax arrays."""
    return type(value)(**torch_to_jax(dataclasses.asdict(value)))


@torch_to_jax.register(torch.device)
def torch_to_jax_device(torch_device: torch.device) -> jax.Device:
    if torch_device.type == "cuda":
        backend = "gpu"
    elif jax.default_backend() == "tpu":
        backend = "tpu"
    else:
        backend = "cpu"
    devices = jax.devices(backend=backend)
    if torch_device.type == "cuda":
        return devices[torch_device.index]
    else:
        # For CPU/TPU, use index if specified, otherwise default to first device
        index = torch_device.index if torch_device.index is not None else 0
        return devices[index]


@torch_to_jax.register(collections.abc.Callable)
def torch_to_jax_callable(torch_callable: Callable) -> Callable:
    """Wraps a torch function so that it can be used from jax.

    NOTE: You shouldn't expect jax.jit or jax.grad to work through this torch function (at least
    for now).
    """
    from .to_torch import jax_to_torch

    @functools.wraps(torch_callable)
    def _wrapped(*jax_args, **jax_kwargs):
        torch_args = [jax_to_torch(arg) for arg in jax_args]
        torch_kwargs = {k: jax_to_torch(v) for k, v in jax_kwargs.items()}
        torch_outputs = torch_callable(*torch_args, **torch_kwargs)
        return torch_to_jax(torch_outputs)

    return _wrapped
