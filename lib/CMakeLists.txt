if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif()

include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

add_subdirectory(OpModel)
add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Scheduler)
add_subdirectory(Support)
add_subdirectory(Transforms)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# MOLA local patch (2026-05-11): split MLIR libs from tt-mlir's own
# libs. When MLIR_LINK_MLIR_DYLIB=ON, MLIR libs link via libMLIR.so
# (single shared MLIR in the process). Only tt-mlir's own static libs
# get --whole-archive'd into libTTMLIRCompiler.so. Without this split,
# both libTTMLIRCompiler.so AND the consuming binary embed MLIR's
# static initializers (pass-pipeline registry, cl::opt enum-values for
# pass options like SparseTensor's LoopOrderingStrategy::kDefault) and
# they collide at startup with "Option 'default' already exists!".
if(MLIR_LINK_MLIR_DYLIB)
  # Partition libs into:
  #   TTMLIR_OWN_LIBS  - tt-mlir's own (names starting with TTMLIR* or
  #                       MLIRTT*). --whole-archive'd into the .so so the
  #                       pass-registration static globals are linked in.
  #   TTMLIR_MLIR_LIBS - MLIR's libs (everything else). Linked normally
  #                       so MLIR_LINK_MLIR_DYLIB routes through libMLIR.so.
  set(_all_libs
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    TTMLIRSupport
    MLIRPass
    MLIRSupport
    MLIRRegisterAllPasses
    MLIRLinalgSynchronizableOpInterfaceImpl)
  set(TTMLIR_OWN_LIBS)
  set(TTMLIR_MLIR_LIBS)
  foreach(_lib ${_all_libs})
    # MLIR's install libs start with MLIR* (and not MLIRTT*). All other
    # patterns are tt-mlir-bundled libs that must --whole-archive into
    # the .so so their pass-pipeline registrations link in.
    if (_lib MATCHES "^MLIR" AND NOT _lib MATCHES "^MLIRTT")
      list(APPEND TTMLIR_MLIR_LIBS ${_lib})
    else()
      list(APPEND TTMLIR_OWN_LIBS ${_lib})
    endif()
  endforeach()
  # Also ensure libMLIR.so is in the .so's link line so transitive
  # MLIR refs resolve via the shared lib at runtime.
  list(APPEND TTMLIR_MLIR_LIBS MLIR)
  set(TTMLIR_LIBS ${TTMLIR_OWN_LIBS})
else()
  set(TTMLIR_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    TTMLIRSupport
    MLIRPass
    MLIRSupport
    MLIRRegisterAllPasses
    MLIRLinalgSynchronizableOpInterfaceImpl
  )
  set(TTMLIR_MLIR_LIBS)
endif()

set(SHARDY_LIBS)
if (TTMLIR_ENABLE_STABLEHLO)
  list(APPEND SHARDY_LIBS
    SdyDialect
    SdyRegister
    SdyTransformsPropagationOpShardingRuleBuilder
    SdyTransformsPropagationPasses
    SdyTransformsPropagationOpShardingRuleRegistry
    SdyTransformsPropagationShardingGroupMap
    SdyTransformsPropagationShardingProjection
    SdyTransformsPropagationAutoPartitionerRegistry
    SdyAxisListRef
    SdyExplicitReshardsUtil
    SdyTransformsPropagationUtils
    SdyTransformsPropagationBasicFactorPropagation
    SdyTransformsPropagationAggressiveFactorPropagation
    SdyTransformsCommonOpProperties
    SdyTransformsCommonShardingWalker
    SdyTransformsPropagationDebuggingSourceSharding
    SdyCommonFileUtils
    SdyTransformsExportPasses
    SdyTransformsImportPasses
    SdyTransformsPasses
  )
endif()

set(SHARDY_HEADER_DIRS)
if (TTMLIR_ENABLE_STABLEHLO)
    list(APPEND SHARDY_HEADER_DIRS
      ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    )
endif()

#
# Static Target for tools like opt/translate/lspserver
#
add_mlir_library(TTMLIRCompilerStatic
  RegisterAll.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARDY_HEADER_DIRS}

  LINK_LIBS PUBLIC
  ${TTMLIR_LIBS}
  coverage_config
  PRIVATE
  ${SHARDY_LIBS}
)

target_include_directories(TTMLIRCompilerStatic
  INTERFACE
    $<BUILD_INTERFACE:${TTMLIR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TTMLIR_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

#
# Shared Target for compiler frontends
#
add_library(TTMLIRCompiler SHARED RegisterAll.cpp)

set_target_properties(TTMLIRCompiler PROPERTIES
  BUILD_RPATH "$ORIGIN:$ORIGIN/../../third_party/tt-metal/src/tt-metal/build/lib"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH FALSE
)

# Forces the inclusion of all symbols in the shared object
# This is necessary because the JIT will not be able to find the symbols otherwise
if (APPLE)
  set(REEXPORT_TTMLIR_LIBS -Wl,-all_load ${TTMLIR_LIBS})
else()
  set(REEXPORT_TTMLIR_LIBS -Wl,--whole-archive ${TTMLIR_LIBS} -Wl,--no-whole-archive)
endif()

target_link_libraries(TTMLIRCompiler PRIVATE
    # MOLA F3 path C (2026-05-11): force libMLIR.so + libLLVM.so to be
    # processed BEFORE any MLIR/LLVM static archive in the link line.
    # Otherwise the linker walks .a files first; their TUs (e.g.
    # LLVMCodeGen's "enable-branch-hint" cl::opt, MLIR SparseTensor's
    # "default" cl::opt) get pulled in to satisfy undef refs, and the
    # static-init constructors fire alongside the .so's already-
    # registered counterparts at process startup → "Option already
    # exists!". Using `--no-as-needed` ensures the linker actually
    # walks the .so's exported symbols rather than dropping them as
    # unreferenced.
    "-Wl,--no-as-needed"
    "${LLVM_LIBRARY_DIR}/libMLIR.so.22.0git"
    "${LLVM_LIBRARY_DIR}/libLLVM.so.22.0git"
    "-Wl,--as-needed"
    ${REEXPORT_TTMLIR_LIBS}
    ${TTMLIR_MLIR_LIBS}
    flatbuffers
)

target_link_libraries(TTMLIRCompiler PUBLIC coverage_config)

target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib64)

install(TARGETS TTMLIRCompiler DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)

if(NOT TTMLIR_ENABLE_STABLEHLO)
  install(TARGETS TTMLIRCompilerStatic
    EXPORT TTMLIRTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT StaticLib)
else()
  # Install without export when StableHLO is enabled since it has external
  # dependencies that are not part of the TTMLIR install.
  install(TARGETS TTMLIRCompilerStatic
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT StaticLib)
endif()

# Add coverage_config to all targets in lib
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target IN ITEMS TTMLIRCompilerStatic TTMLIRCompiler)
    if (TARGET ${target})
      target_link_libraries(${target} INTERFACE coverage_config)
    endif()
  endforeach()
endif()
