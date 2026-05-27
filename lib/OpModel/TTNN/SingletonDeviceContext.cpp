// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"

#include <cstdlib>
#include <string>

namespace mlir::tt::ttnn::op_model {

SingletonDeviceContext::~SingletonDeviceContext() {
  assert(
      m_device == nullptr &&
      "Device should be null when SingletonDeviceContext is destructed. Call "
      "closeInstance() once you are done with the device.");
}

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance = SingletonDeviceContext();

  return instance;
}

void SingletonDeviceContext::resetInstance() {
  SingletonDeviceContext &instance = getInstance();
  assert(!instance.m_isExternalDevice &&
         "Cannot reset instance when using an external device.");
  bool wasMock = instance.m_isMockDevice;
  instance.closeInstance();
  instance.openDevice(::tt::constants::opModelDefaultTraceRegionSize, wasMock);
}

void SingletonDeviceContext::closeInstance() {
  SingletonDeviceContext &instance = getInstance();
  assert(instance.m_device != nullptr && "No device to close");
  bool wasExternalDevice = instance.m_isExternalDevice;
  bool wasMockDevice = instance.m_isMockDevice;
  instance.m_device.reset();
  instance.m_isMockDevice = false;
  if (!wasExternalDevice && wasMockDevice) {
    ::tt::tt_metal::experimental::disable_mock_mode();
  }
}

void SingletonDeviceContext::setExternalDevice(
    std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> device) {
  SingletonDeviceContext &instance = getInstance();
  assert(device != nullptr && "External device pointer cannot be null");
  assert(instance.m_device == nullptr &&
         "Device is already initialized. Cannot set external device.");
  instance.m_device = std::move(device);
  instance.m_isExternalDevice = true;
}

void SingletonDeviceContext::setSystemDesc(ttcore::SystemDescAttr systemDesc) {
  SingletonDeviceContext &instance = getInstance();
  instance.m_systemDesc = systemDesc;
}

void SingletonDeviceContext::openMockDevice(
    const size_t traceRegionSize,
    const std::optional<std::pair<size_t, size_t>> &meshShape) {
#ifdef TTMLIR_DISABLE_MOCK_DEVICE
  bool disableMock = true;
#else
  bool disableMock = false;
#endif
  if (const char *env = std::getenv("TTMLIR_DISABLE_MOCK_DEVICE")) {
    disableMock = std::string(env) != "0";
  }
  openDevice(traceRegionSize, /*isMock=*/!disableMock, meshShape);
}

void SingletonDeviceContext::openDevice(
    const size_t traceRegionSize, bool isMock,
    const std::optional<std::pair<size_t, size_t>> &meshShape) {
  assert(m_device == nullptr &&
         "Device is already initialized. Cannot open device again.");

  m_isMockDevice = isMock;

  if (isMock) {
    assert(m_systemDesc && "System desc must be set for mock device mode");
    auto arch = m_systemDesc.getChipDesc(0).getArch().getValue();
    uint32_t numChips = m_systemDesc.getChipDescIndices().size();
    ::tt::ARCH metalArch;
    switch (arch) {
    case ttcore::Arch::WormholeB0:
      metalArch = ::tt::ARCH::WORMHOLE_B0;
      break;
    case ttcore::Arch::Blackhole:
      metalArch = ::tt::ARCH::BLACKHOLE;
      break;
    case ttcore::Arch::Quasar:
      metalArch = ::tt::ARCH::QUASAR;
      break;
    }
    // 6U Wormhole Galaxy chips report compute_with_storage_grid_size (x=7,
    // y=10), distinguishing them from T3K Wormhole chips (x=7, y=8). Mock
    // mode keys its cluster-descriptor lookup on (arch, num_chips) — for
    // num_chips in {1, 2, 4, 8} on Wormhole it picks single-Wormhole or T3K
    // cluster descs, neither of which matches a 6U Galaxy chip's grid. When
    // the captured system_desc shows 6U Galaxy chips (grid y == 10), force
    // num_chips=32 in mock mode so it picks 6u_cluster_desc.yaml; the
    // caller-supplied mesh_shape opens a submesh view of the 32-chip mock
    // cluster (see mock_device_util.cpp's "case 16" comment for the same
    // pattern with the 1x16 torus topology).
    uint32_t mockNumChips = numChips;
    if (metalArch == ::tt::ARCH::WORMHOLE_B0) {
      auto grid = m_systemDesc.getChipDesc(0).getGrid();
      if (grid.size() == 2 && grid[0] == 10 && numChips < 32) {
        mockNumChips = 32;
      }
    }
    ::tt::tt_metal::experimental::configure_mock_mode(metalArch, mockNumChips);
  }

  // todo: this replicates logic in
  // runtime/include/tt/runtime/detail/common/common.h, move to shared location
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;
  // Mirror runtime/lib/ttnn/runtime.cpp / runtime/lib/common/system_desc.cpp:
  // moe_compute requires DispatchCoreAxis::COL. ETH + COL is rejected by
  // tt-metal's Python wrapper and produces an invalid DRAM-bank-to-worker
  // mapping; coerce to WORKER so the OpModel mock device's compute grid
  // matches the runtime mesh device the produced binary will run on.
  if (dispatchCoreType == ::tt::tt_metal::DispatchCoreType::ETH) {
    dispatchCoreType = ::tt::tt_metal::DispatchCoreType::WORKER;
  }
  ::tt::tt_metal::DispatchCoreConfig dispatchCoreConfig(
      dispatchCoreType, ::tt::tt_metal::DispatchCoreAxis::COL);

  ::tt::tt_metal::distributed::MeshShape shape{
      meshShape ? static_cast<unsigned int>(meshShape->first) : 1,
      meshShape ? static_cast<unsigned int>(meshShape->second) : 1};
  m_device = ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /* num_hw_cqs = */ 1, dispatchCoreConfig);

  m_device->disable_and_clear_program_cache();
}

void SingletonDeviceContext::reshapeMeshDevice(
    const std::pair<size_t, size_t> &meshShape, size_t traceRegionSize) {
  assert(m_device != nullptr && "Device must be initialized to reshape");
  assert(m_isMockDevice && "Can only reshape mock devices");

  m_device.reset();

  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;
  if (dispatchCoreType == ::tt::tt_metal::DispatchCoreType::ETH) {
    dispatchCoreType = ::tt::tt_metal::DispatchCoreType::WORKER;
  }
  ::tt::tt_metal::DispatchCoreConfig dispatchCoreConfig(
      dispatchCoreType, ::tt::tt_metal::DispatchCoreAxis::COL);

  ::tt::tt_metal::distributed::MeshShape shape{
      static_cast<unsigned int>(meshShape.first),
      static_cast<unsigned int>(meshShape.second)};
  m_device = ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /* num_hw_cqs = */ 1, dispatchCoreConfig);

  m_device->disable_and_clear_program_cache();
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
