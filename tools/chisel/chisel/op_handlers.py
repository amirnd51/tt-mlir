# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.runtime import TensorRef

from golden import GoldenMapTensor

from .ops import get_op_inputs, get_op_outputs
from .report import (
    ChiselRecord,
    GoldenEvictedPayload,
    GoldenPromotedPayload,
    GoldenPromotionSource,
    NumericsMode,
    SkippedNumericsPayload,
)
from .safety import chisel_safe
from .utils import retrieve_tensor
from .validators import check_numerics, check_shape_dtype


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


def _publish_to_program_pool(
    ctx, tensor_ref: TensorRef, golden: GoldenMapTensor
) -> None:
    """Write `golden` into the cross-program pool keyed by the stored runtime
    tensor's globalId, and register a destroy-callback evictor so the entry
    dies with the underlying TTNNTensorWrapper."""
    rt_program_context = ctx.rt_program_context
    global_id = tt_runtime.get_tensor_global_id_from_pool(
        rt_program_context, tensor_ref
    )
    if global_id is None:
        return

    program_io_pool = ctx.program_io_pool
    program_io_pool[global_id] = golden

    # Capture only the gid and the pool dict so the destroy closure has no
    # references back to ChiselContext (which might have outlived the session).
    def _evict(
        _gid: int = global_id,
        _pool: Dict[int, GoldenMapTensor] = program_io_pool,
    ) -> None:
        _pool.pop(_gid, None)

    tt_runtime.register_pool_tensor_destroy_callback(
        rt_program_context, tensor_ref, _evict
    )


def _lookup_from_program_pool(
    ctx, tensor_ref: TensorRef
) -> Optional[GoldenMapTensor]:
    """Return the cross-program golden for the given tensor_ref, or None."""
    global_id = tt_runtime.get_tensor_global_id_from_pool(
        ctx.rt_program_context, tensor_ref
    )
    if global_id is None:
        return None
    return ctx.program_io_pool.get(global_id)


@chisel_safe
def _subprogram_pre_op(ctx, config) -> None:
    """For func.call / ttcore.LoadCachedOp: publish each parent input's
    accumulated golden into the cross-program pool keyed by the input
    Tensor's globalId, so the sub-program's default pre-op finds it via
    the standard function-arg cross-pool lookup. No-op when accumulation
    is off."""
    if not ctx.checks_config.accumulation:
        return

    op = ctx.op
    pool = ctx.golden_tensor_pool
    asm_state = ctx.asm_state
    for mlir_input, rt_tensor_ref in zip(
        get_op_inputs(op), ctx.input_refs, strict=True
    ):
        check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_input, rt_tensor_ref)
        parent_ssa = mlir_input.get_name(asm_state)
        golden = pool.get(parent_ssa)
        if golden is None:
            # Upstream op was no_golden or the SSA was never seeded; the
            # sub-program will fall back to source=device for this arg.
            continue
        _publish_to_program_pool(ctx, rt_tensor_ref, golden)


@chisel_safe
def _subprogram_post_op(ctx, config) -> None:
    """For func.call / ttcore.LoadCachedOp: install each output's accumulated
    golden (published by the sub-program's post-op, by globalId) into the
    parent's golden pool and PCC-check it against the device tensor. On a
    pool miss (e.g. LoadCachedOp cache hit in a fresh session), degrade to
    seed-from-device. No-op when accumulation is off."""
    if not ctx.checks_config.accumulation:
        return

    op = ctx.op
    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool

    for mlir_output, output_ref in zip(
        get_op_outputs(op), ctx.output_refs, strict=True
    ):
        check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_output, output_ref)
        device_tensor = retrieve_tensor(ctx.rt_program_context, output_ref)
        check_shape_dtype(op, "mlir_vs_runtime_tensor", mlir_output, device_tensor)
        ssa = mlir_output.get_name(asm_state)

        golden = _lookup_from_program_pool(ctx, output_ref)
        if golden is not None:
            check_shape_dtype(op, "mlir_vs_golden", mlir_output, golden)
            pool[ssa] = golden
            source = GoldenPromotionSource.PROGRAM_POOL
        else:
            pool[ssa] = device_tensor
            source = GoldenPromotionSource.DEVICE

        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_promoted",
                ssa=ssa,
                payload=GoldenPromotedPayload(source=source),
            )
        )

        if golden is None:
            # No golden to compare against; chain restarts here.
            continue

        if config.skip_pcc:
            ctx.write_record(
                ChiselRecord(
                    op=op.name,
                    check="numerics",
                    ssa=ssa,
                    payload=SkippedNumericsPayload(mode=NumericsMode.ACCUMULATED),
                )
            )
            continue

        check_numerics(
            ctx, op, ssa, golden, device_tensor, mode=NumericsMode.ACCUMULATED
        )
