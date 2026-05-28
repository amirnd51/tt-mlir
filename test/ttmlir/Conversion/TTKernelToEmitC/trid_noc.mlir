// RUN: ttmlir-opt --convert-ttkernel-to-emitc --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func @trid_read_barrier_path
// CHECK: emitc.verbatim "Noc noc0(0);"
// CHECK-NEXT: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT: emitc.verbatim "noc0.async_read_barrier<Noc::BarrierMode::TXN_ID>({});" args %[[TRID]] : i32
func.func @trid_read_barrier_path() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_idx = arith.constant 0 : i8
  "ttkernel.noc_async_read_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  return
}

// -----

// CHECK-LABEL: func @trid_write_path
// CHECK: emitc.verbatim "UnicastEndpoint unicast_ep;"
// CHECK: emitc.verbatim "Noc noc0(0);"
// CHECK-NEXT: %[[TRID:.*]] = "emitc.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT: %[[NOC_IDX:.*]] = "emitc.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT: %[[X:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[Y:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
// CHECK-NEXT: %[[SRC:.*]] = "emitc.constant"() <{value = 256 : i32}> : () -> i32
// CHECK-NEXT: %[[DST:.*]] = "emitc.constant"() <{value = 512 : i32}> : () -> i32
// CHECK-NEXT: %[[SIZE:.*]] = "emitc.constant"() <{value = 128 : i32}> : () -> i32
// CHECK-NEXT: emitc.verbatim "noc0.async_write<Noc::TxnIdMode::ENABLED, Noc::ResponseMode::NON_POSTED, NOC_MAX_BURST_SIZE>
// CHECK-SAME: .noc_x = {}, .noc_y = {}, .addr = static_cast<uint32_t>
// CHECK-SAME: args %[[SRC]], %[[SIZE]], %[[X]], %[[Y]], %[[DST]], %[[TRID]]
// CHECK-NEXT: emitc.verbatim "noc0.async_write_barrier<Noc::BarrierMode::TXN_ID>({});" args %[[TRID]] : i32
func.func @trid_write_path() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %trid = arith.constant 3 : i32
  %noc_idx = arith.constant 0 : i8
  %x = arith.constant 0 : index
  %y = arith.constant 1 : index
  %src = arith.constant 256 : i32
  %dst = arith.constant 512 : i32
  %size = arith.constant 128 : i32
  ttkernel.noc_async_write_one_packet_with_trid(%src, core[%x, %y], %dst, %size, %trid, %noc_idx) : (i32, index, index, i32, i32, i32, i8) -> ()
  "ttkernel.noc_async_write_barrier_with_trid"(%trid, %noc_idx) : (i32, i8) -> ()
  return
}
