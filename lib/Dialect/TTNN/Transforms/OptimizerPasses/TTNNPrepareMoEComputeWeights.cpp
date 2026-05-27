// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPAREMOECOMPUTEWEIGHTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNPrepareMoEComputeWeights
    : public impl::TTNNPrepareMoEComputeWeightsBase<
          TTNNPrepareMoEComputeWeights> {
public:
  using impl::TTNNPrepareMoEComputeWeightsBase<
      TTNNPrepareMoEComputeWeights>::TTNNPrepareMoEComputeWeightsBase;

  // If a prep result feeds directly into a func.return, the enclosing
  // function's result type must be refreshed to match the refined op result
  // type. (Other consumers read the type through use-def so they don't need
  // an explicit fix-up.)
  void refreshEnclosingFuncReturnType(mlir::Value refinedResult) {
    for (mlir::OpOperand &use : refinedResult.getUses()) {
      auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(use.getOwner());
      if (!retOp) {
        continue;
      }
      auto funcOp = retOp->getParentOfType<mlir::func::FuncOp>();
      if (!funcOp) {
        continue;
      }
      llvm::SmallVector<mlir::Type> newResultTypes(
          funcOp.getFunctionType().getResults().begin(),
          funcOp.getFunctionType().getResults().end());
      newResultTypes[use.getOperandNumber()] = refinedResult.getType();
      funcOp.setFunctionType(mlir::FunctionType::get(
          funcOp.getContext(), funcOp.getFunctionType().getInputs(),
          newResultTypes));
    }
  }

  // For every ttnn.prepare_moe_compute_w0_w1_weights / _w2_weights op, replace
  // the placeholder result type (which inherits the TTIR-side declaration)
  // with the spec returned by OpModel via graph-captured invocation of the
  // shared C++ helper. The downstream ttnn.moe_compute consumer reads the
  // refined types directly through use-def.
  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal("TTNNPrepareMoEComputeWeights requires "
                                    "OpModel support to be enabled.");
#else
    op_model::ScopedSingletonDeviceGuard deviceGuard(getOperation());
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](ttnn::PrepareMoEComputeW0W1WeightsOp op) {
      std::optional<mlir::RankedTensorType> bias0Type;
      if (op.getBias_0()) {
        bias0Type = op.getBias_0().getType();
      }
      std::optional<mlir::RankedTensorType> bias1Type;
      if (op.getBias_1()) {
        bias1Type = op.getBias_1().getType();
      }
      std::optional<ttnn::MemoryConfigAttr> outMemCfg;
      if (op.getOutputMemoryConfigAttr()) {
        outMemCfg = op.getOutputMemoryConfigAttr();
      }
      mlir::RankedTensorType refined =
          op_model::getPreparedMoEComputeW0W1WeightsOutputType(
              op.getOperation(), op.getW0().getType(), op.getW1().getType(),
              bias0Type, bias1Type, op.getHiddenSize(),
              op.getIntermediateSize(), outMemCfg);
      op.getResult().setType(refined);
      refreshEnclosingFuncReturnType(op.getResult());
    });

    moduleOp.walk([&](ttnn::PrepareMoEComputeW2WeightsOp op) {
      std::optional<mlir::RankedTensorType> bias2Type;
      if (op.getBias_2()) {
        bias2Type = op.getBias_2().getType();
      }
      std::optional<ttnn::MemoryConfigAttr> outMemCfg;
      if (op.getOutputMemoryConfigAttr()) {
        outMemCfg = op.getOutputMemoryConfigAttr();
      }
      mlir::RankedTensorType refined =
          op_model::getPreparedMoEComputeW2WeightsOutputType(
              op.getOperation(), op.getW2().getType(), bias2Type,
              op.getHiddenSize(), op.getIntermediateSize(), outMemCfg);
      op.getResult().setType(refined);
      refreshEnclosingFuncReturnType(op.getResult());
    });
#endif // TTMLIR_ENABLE_OPMODEL
  }
};

} // namespace mlir::tt::ttnn
