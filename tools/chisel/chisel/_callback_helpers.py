# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared between callbacks.py (default handlers) and op_handlers.py
(per-op overrides). Keep this module free of imports from callbacks.py /
op_handlers.py so neither side has a cycle."""
from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.runtime import TensorRef
from ttmlir.ir import Value

from golden import GoldenMapTensor

from .exceptions import IrRuntimeMismatch
from .ops import SSAName
from .report import ChiselRecord, NumericsMode, SkippedNumericsPayload
from .utils import get_op_asm, retrieve_tensor
from .validators import check_numerics, check_shape_dtype


def assert_op_matches_runtime(ctx: "ChiselContext") -> None:
    """Raise IrRuntimeMismatch if chisel and the runtime point at different ops."""
    rt_debug = tt_runtime.get_op_debug_str(ctx.rt_op_context)
    op = ctx.op
    if rt_debug.strip() == get_op_asm(op).strip():
        return
    raise IrRuntimeMismatch(op, "ir_vs_runtime_op", rt_debug)


def validate_and_retrieve_tensor(
    ctx: "ChiselContext", mlir_value: Value, rt_tensor_ref: TensorRef
) -> GoldenMapTensor:
    op = ctx.op
    check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_value, rt_tensor_ref)
    tensor = retrieve_tensor(ctx.rt_program_context, rt_tensor_ref)
    check_shape_dtype(op, "mlir_vs_runtime_tensor", mlir_value, tensor)
    return tensor


def emit_pcc(
    ctx: "ChiselContext",
    op,
    ssa: SSAName,
    mlir_output: Value,
    golden_out: GoldenMapTensor,
    device_tensor: GoldenMapTensor,
    *,
    mode: NumericsMode,
    skip_pcc: bool,
) -> None:
    """Shape/dtype + PCC for one (golden, device) pair under `mode`."""
    check_shape_dtype(op, "mlir_vs_golden", mlir_output, golden_out)
    if skip_pcc:
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="numerics",
                ssa=ssa,
                payload=SkippedNumericsPayload(mode=mode),
            )
        )
        return
    check_numerics(ctx, op, ssa, golden_out, device_tensor, mode=mode)
