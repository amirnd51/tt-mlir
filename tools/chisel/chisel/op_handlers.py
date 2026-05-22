# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from _ttmlir_runtime import runtime as tt_runtime

from golden import GoldenMapTensor

from ._callback_helpers import emit_pcc, validate_and_retrieve_tensor
from .ops import get_op_inputs, get_op_outputs, is_cpu_hoist_call
from .report import (
    ChiselRecord,
    GoldenEvictedPayload,
    GoldenPromotedPayload,
    NoGoldenPayload,
    NumericsMode,
)
from .safety import chisel_safe
from .utils import golden_to_runtime_tensor, runtime_to_golden_tensor


@chisel_safe
def _deallocate_pre_op(ctx, config) -> None:
    """Evict each input SSA from the golden pool; emit one record per eviction."""
    pool = ctx.golden_tensor_pool
    asm_state = ctx.asm_state
    op = ctx.op
    for inp in get_op_inputs(op):
        ssa = inp.get_name(asm_state)
        if pool.pop(ssa, None) is None:
            continue

        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_evicted",
                ssa=ssa,
                payload=GoldenEvictedPayload(),
            )
        )


@chisel_safe
def _noop_post_op(ctx, config) -> None:
    pass


def _invoke_cpu_op(ctx, inputs: List[GoldenMapTensor]) -> List[GoldenMapTensor]:
    """Re-invoke the dylib behind the current CpuOp with `inputs` as goldens."""
    rt_inputs = [golden_to_runtime_tensor(gmt) for gmt in inputs]
    rt_outputs = tt_runtime.invoke_cpu_op(
        ctx.rt_program_context, ctx.rt_op_context, rt_inputs
    )
    return [runtime_to_golden_tensor(rt) for rt in rt_outputs]


@chisel_safe
def _cpu_hoist_pre_op(ctx, config) -> None:
    """func.CallOp PRE.

    For CPU-hoisted calls, promote any input SSA not already in the golden pool
    by seeding it from the live device tensor (typically a function arg). For
    non-hoisted func.CallOps, preserve the previous no_golden behavior.
    """
    op = ctx.op
    if not is_cpu_hoist_call(op):
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_not_implemented",
                payload=NoGoldenPayload(),
            )
        )
        return

    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool
    for mlir_input, rt_ref in zip(get_op_inputs(op), ctx.input_refs, strict=True):
        ssa = mlir_input.get_name(asm_state)
        if ssa in pool:
            continue
        tensor = validate_and_retrieve_tensor(ctx, mlir_input, rt_ref)
        pool[ssa] = tensor
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_promoted",
                ssa=ssa,
                payload=GoldenPromotedPayload(),
            )
        )


@chisel_safe
def _cpu_hoist_post_op(ctx, config) -> None:
    """func.CallOp POST.

    For CPU-hoisted calls, invoke the dylib with accumulated goldens for the
    input SSAs, seed each output SSA into the golden pool, and PCC-check each
    output against the live device tensor. For non-hoisted calls this is a
    no-op (matching the previous no_golden behavior).
    """
    op = ctx.op
    if not is_cpu_hoist_call(op):
        return
    if not ctx.checks_config.accumulation:
        return

    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool
    output_vals = get_op_outputs(op)
    output_ssas = [v.get_name(asm_state) for v in output_vals]

    input_goldens = [pool[v.get_name(asm_state)] for v in get_op_inputs(op)]
    accum_outs = _invoke_cpu_op(ctx, input_goldens)

    for ssa, gmt in zip(output_ssas, accum_outs, strict=True):
        pool[ssa] = gmt

    for mlir_out, out_ref, ssa, accum_out in zip(
        output_vals, ctx.output_refs, output_ssas, accum_outs, strict=True
    ):
        device_gmt = validate_and_retrieve_tensor(ctx, mlir_out, out_ref)
        emit_pcc(
            ctx,
            op,
            ssa,
            mlir_out,
            accum_out,
            device_gmt,
            mode=NumericsMode.ACCUMULATED,
            skip_pcc=config.skip_pcc,
        )
