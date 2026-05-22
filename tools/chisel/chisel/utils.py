# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
import os
import traceback
from typing import Optional, Tuple

import torch

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, Tensor, TensorRef

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype

logger = logging.getLogger("chisel")


def get_torch_tensor(tensor: Tensor) -> torch.Tensor:
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = mlir_datatype_to_torch_dtype(rt_dtype)
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape)


# Inverse of mlir_datatype_to_torch_dtype. Kept local so chisel does not pull
# in builder_runtime's copy. Move to golden/mapping.py if a second consumer
# appears.
_TORCH_TO_RUNTIME_DTYPE = {
    torch.float32: tt_runtime.DataType.Float32,
    torch.float16: tt_runtime.DataType.Float16,
    torch.bfloat16: tt_runtime.DataType.BFloat16,
    torch.float64: tt_runtime.DataType.Float64,
    torch.int8: tt_runtime.DataType.Int8,
    torch.int16: tt_runtime.DataType.Int16,
    torch.int32: tt_runtime.DataType.Int32,
    torch.int64: tt_runtime.DataType.Int64,
    torch.uint8: tt_runtime.DataType.UInt8,
    torch.uint16: tt_runtime.DataType.UInt16,
    torch.uint32: tt_runtime.DataType.UInt32,
    torch.uint64: tt_runtime.DataType.UInt64,
    torch.bool: tt_runtime.DataType.Bool,
}


def torch_dtype_to_runtime_dtype(dtype: torch.dtype):
    try:
        return _TORCH_TO_RUNTIME_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"no runtime DataType for torch dtype {dtype}")


def retrieve_tensor(
    rt_program_context: CallbackContext, rt_tensor_ref: TensorRef
) -> GoldenMapTensor:
    device_tensor = tt_runtime.retrieve_tensor_from_pool(
        rt_program_context, rt_tensor_ref
    )
    if device_tensor is None:
        raise RuntimeError(
            "retrieve_tensor_from_pool returned no tensor for the requested ref"
        )
    return GoldenMapTensor({0: get_torch_tensor(device_tensor)}, (1, 1))


def golden_to_runtime_tensor(gmt: GoldenMapTensor) -> Tensor:
    """Build a host runtime.Tensor from a GoldenMapTensor for invoke_cpu_op.

    Single-chip only today (mesh_shape == (1, 1)); multichip support lands with
    the multichip-tensors-chisel branch via create_multi_device_host_tensor.
    """
    if gmt.mesh_shape != (1, 1):
        raise NotImplementedError(
            f"golden_to_runtime_tensor: multichip mesh_shape {gmt.mesh_shape} "
            "not yet supported"
        )
    t = gmt.shard_map[0].contiguous()
    return tt_runtime.create_owned_host_tensor(
        t.data_ptr(),
        list(t.shape),
        list(t.stride()),
        t.element_size(),
        torch_dtype_to_runtime_dtype(t.dtype),
    )


def runtime_to_golden_tensor(rt: Tensor) -> GoldenMapTensor:
    """Wrap a host runtime.Tensor as a single-shard GoldenMapTensor."""
    return GoldenMapTensor({0: get_torch_tensor(rt)}, (1, 1))


def get_op_asm(op) -> str:
    # Mirror OpPrintingFlags from FuncOpToProgram so the asm string matches
    # the flatbuffer debug_info.
    return op.get_asm(
        enable_debug_info=True,
        large_elements_limit=16,
        large_resource_limit=64,
        skip_regions=True,
        assume_verified=True,
    ).strip()


def debug_wrap(fn):
    # Checked per-call so exporting CHISEL_DEBUG mid-session takes effect
    # without re-import.

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if os.environ.get("CHISEL_DEBUG"):
                import pdb

                traceback.print_exc()
                pdb.post_mortem()
            raise

    return wrapper
