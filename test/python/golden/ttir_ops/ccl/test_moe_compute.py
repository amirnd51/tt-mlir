# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile-only smoke test for the fused ttir.moe_compute op + weight prep.

Generates a flatbuffer on a (4, 8) Wormhole 6U Galaxy mesh (32 chips) that chains
``prepare_moe_compute_w0_w1_weights`` and ``prepare_moe_compute_w2_weights``
into ``moe_compute``. The compiler must run through:

  * TTIR -> TTNN conversion (covered by the lit test as well)
  * Workarounds (HEIGHT_SHARDED outputs 0/3/4 + dtype/layout factory)
  * ``TTNNAllocateDistributedOpBuffers`` (binds optional_output_tensor)
  * ``TTNNAllocateDistributedOpSemaphores`` (binds cross_device_semaphore)
  * Flatbuffer serialization

``skip_exec=True`` because validating the kernel needs a Galaxy and reference
weights from tt-metal's prepare_w0_w1_tensor_for_moe_compute. The artifact
this test produces is the input you'd feed ``ttrt run`` on a Galaxy.
"""

import pytest
import torch
from collections import OrderedDict
from typing import List, Optional

import _ttmlir_runtime as tt_runtime
from builder.base.builder_utils import DeferredDevice, Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


# Shape constants chosen so prepare_w0_w1 / _w2 hit the smallest non-degenerate
# tile/block configuration: one BLOCK_TILES_H K-block (7 * 32 = 224) and a
# single groups_per_core for W0/W1. The exact post-prep result shapes are
# kernel-dependent (num_cores comes from the device's DRAM bank layout), so
# this test relies on the OpModel/builder-provided result types; the runtime
# entry point will assert if they disagree.

L = 1  # layers
E_LOCAL = 4  # experts per device
H = 256  # hidden_size (must be a multiple of TILE=32)
# intermediate_size in tiles must be >= num matmul-ring cores (12 on WH);
# 384/32 = 12 tiles is the minimum.
N_INTER = 384  # intermediate_size
BD = 128  # B*D dispatched tokens per device
K = 4  # top-k
NUM_CORES = 12  # WH DRAM banks per device
GROUPS_PER_CORE = 1
K_PADDED = 448  # ceil(H/TILE/BLOCK_TILES_H) * TILE * BLOCK_TILES_H, H=256 -> 2*7*32


@pytest.mark.parametrize("mesh_shape", [(1, 8)], ids=["1x8"])
@pytest.mark.parametrize(
    "fabric_config",
    [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING],
    ids=["fabric_1d_ring"],
)
def test_moe_compute_smoke(mesh_shape, fabric_config, request):
    """Compile-only smoke test. Use ``ttrt run`` on the resulting flatbuffer.

    Mirrors tt-metal's `test_moe_compute_6U.py` device-params:
    DispatchCoreAxis=COL (forces WORKER dispatch in runtime; ETH+COL is
    rejected by tt-metal), reliability_mode=RELAXED_INIT, FABRIC_1D_RING.
    The dispatch config and reliability mode are applied by the tt-mlir
    runtime when opening the mesh device (`runtime/lib/ttnn/runtime.cpp`).
    """

    w0_shape = (L, E_LOCAL, H, N_INTER)
    w1_shape = (L, E_LOCAL, H, N_INTER)
    w2_shape = (L, E_LOCAL, N_INTER, H)

    w0_w1_prep_shape = (
        NUM_CORES,
        L,
        E_LOCAL,
        GROUPS_PER_CORE,
        K_PADDED,
        4 * 32,
    )
    w2_prep_shape = (
        NUM_CORES,
        L,
        E_LOCAL,
        GROUPS_PER_CORE,
        K_PADDED,
        4 * 32,
    )

    tilize_input_shape = (1, 1, BD, H)
    tilize_idx_shape = (1, 1, BD, K)
    tilize_scores_shape = (1, 1, BD, K)
    tilize_mapping_shape = (1, 1, E_LOCAL, K)

    # MoeComputeOp result shapes (declared by the frontend; workarounds reshape
    # outputs 0/3/4 to HEIGHT_SHARDED on workers post-conversion).
    out_per_expert_total_tokens = (1, 1, E_LOCAL, K)
    out_expert_activation = (1, 1, BD, K)
    out_expert_to_token = (1, 1, BD, K)
    out_tilize = (1, 1, BD, H)
    out_matmul = (1, 1, BD, H)
    out_combine = (1, 1, BD, H)

    def module(builder: TTIRBuilder):
        @builder.func(
            [
                w0_shape,
                w1_shape,
                w2_shape,
                tilize_input_shape,
                tilize_idx_shape,
                tilize_scores_shape,
                tilize_mapping_shape,
            ],
            [
                torch.bfloat16,  # w0
                torch.bfloat16,  # w1
                torch.bfloat16,  # w2
                torch.bfloat16,  # tilize_input
                torch.int64,  # tilize_indices
                torch.bfloat16,  # tilize_scores
                torch.int64,  # tilize_mapping
            ],
        )
        def moe_compute_pipeline(
            w0: Operand,
            w1: Operand,
            w2: Operand,
            tilize_input: Operand,
            tilize_idx: Operand,
            tilize_scores: Operand,
            tilize_mapping: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            w0_w1_prep = builder.prepare_moe_compute_w0_w1_weights(
                w0,
                w1,
                hidden_size=H,
                intermediate_size=N_INTER,
                result_shape=w0_w1_prep_shape,
                result_type=torch.bfloat16,
            )
            w2_prep = builder.prepare_moe_compute_w2_weights(
                w2,
                hidden_size=H,
                intermediate_size=N_INTER,
                result_shape=w2_prep_shape,
                result_type=torch.bfloat16,
            )
            results = builder.moe_compute(
                tilize_input,
                tilize_idx,
                tilize_scores,
                tilize_mapping,
                w0_w1_prep,
                w2_prep,
                layer_id=0,
                output_height_shard_dim=4,
                intermediate_size=N_INTER,
                has_bias=False,
                cluster_axis=1,
                activation_function="silu",
                output_shapes=[
                    out_per_expert_total_tokens,
                    out_expert_activation,
                    out_expert_to_token,
                    out_tilize,
                    out_matmul,
                    out_combine,
                ],
                output_types=[
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                ],
            )
            return results[5]  # combine_output

    # moe_compute requires the optimizer to be enabled so its MoE weight-prep
    # pass refines the prep-op result types to DRAM HEIGHT_SHARDED before
    # workarounds / distributed-buffer allocation see them. Use a deferred
    # device so OpModel's mock device can be opened during compilation
    # without conflicting with the real device used for execution.
    compile_and_execute_ttir(
        module,
        target="ttnn",
        mesh_name="mesh",
        device=DeferredDevice(request),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=["enable-optimizer=true"],
        skip_exec=True,
        check_pcc=False,
        disable_golden=True,
        **get_request_kwargs(request),
    )
