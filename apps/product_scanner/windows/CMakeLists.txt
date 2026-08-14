# Project-level configuration.
cmake_minimum_required(VERSION 3.19)
project(product_scanner LANGUAGES CXX)

# The name of the executable created for the application. Change this to change
# the on-disk name of your application.
set(BINARY_NAME "product_scanner")

# Explicitly opt in to modern CMake behaviors to avoid warnings with recent
# versions of CMake.
cmake_policy(VERSION 3.14...3.25)

# Define build configuration option.
get_property(IS_MULTICONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTICONFIG)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Profile;Release"
    CACHE STRING "" FORCE)
else()
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE
      STRING "Flutter build mode" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Profile" "Release")
  endif()
endif()
# Define settings for the Profile build mode.
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_PROFILE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_PROFILE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE}")

# Use Unicode for all projects.
add_definitions(-DUNICODE -D_UNICODE)

# Compilation settings that should be applied to most targets.
#
# Be cautious about adding new options here, as plugins use this function by
# default. In most cases, you should add new options to specific targets instead
# of modifying this function.
function(APPLY_STANDARD_SETTINGS TARGET)
  target_compile_features(${TARGET} PUBLIC cxx_std_17)
  target_compile_options(${TARGET} PRIVATE /W4 /WX /wd"4100")
  target_compile_options(${TARGET} PRIVATE /EHsc)
  target_compile_definitions(${TARGET} PRIVATE "_HAS_EXCEPTIONS=0")
  target_compile_definitions(${TARGET} PRIVATE "$<$<CONFIG:Debug>:_DEBUG>")
endfunction()

# Flutter library and tool build rules.
set(FLUTTER_MANAGED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/flutter")
add_subdirectory(${FLUTTER_MANAGED_DIR})

# Application build; see runner/CMakeLists.txt.
add_subdirectory("runner")


# Generated plugin build rules, which manage building the plugins and adding
# them to the application.
include(flutter/generated_plugins.cmake)


# === Installation ===
# Support files are copied into place next to the executable, so that it can
# run in place. This is done instead of making a separate bundle (as on Linux)
# so that building and running from within Visual Studio will work.
set(BUILD_BUNDLE_DIR "$<TARGET_FILE_DIR:${BINARY_NAME}>")
set(SCANNER_RELEASE_BUILD FALSE)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR FLUTTER_BUILD_MODE STREQUAL "release")
  set(SCANNER_RELEASE_BUILD TRUE)
endif()

# A release composition is the only source of package identity for distributable builds.
set(SCANNER_RELEASE_COMPOSITION "" CACHE FILEPATH
  "BIXOLON release composition JSON")
if(NOT SCANNER_RELEASE_COMPOSITION AND DEFINED ENV{SCANNER_RELEASE_COMPOSITION})
  file(TO_CMAKE_PATH "$ENV{SCANNER_RELEASE_COMPOSITION}"
    SCANNER_RELEASE_COMPOSITION)
endif()
if(SCANNER_RELEASE_COMPOSITION)
  if(NOT EXISTS "${SCANNER_RELEASE_COMPOSITION}")
    message(FATAL_ERROR
      "Release composition does not exist: ${SCANNER_RELEASE_COMPOSITION}")
  endif()
  file(READ "${SCANNER_RELEASE_COMPOSITION}" SCANNER_RELEASE_JSON)
  string(JSON SCANNER_COMPOSED_PACKAGE_PATH GET
    "${SCANNER_RELEASE_JSON}" model_package path)
  get_filename_component(SCANNER_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
  get_filename_component(SCANNER_MODEL_PACKAGE_DIR
    "${SCANNER_REPOSITORY_ROOT}/${SCANNER_COMPOSED_PACKAGE_PATH}" ABSOLUTE)
endif()
# Make the "install" step default, as it's required to run.
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${BUILD_BUNDLE_DIR}" CACHE PATH "..." FORCE)
endif()

set(INSTALL_BUNDLE_DATA_DIR "${CMAKE_INSTALL_PREFIX}/data")
set(INSTALL_BUNDLE_LIB_DIR "${CMAKE_INSTALL_PREFIX}")

install(TARGETS ${BINARY_NAME} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}"
  COMPONENT Runtime)

install(FILES "${FLUTTER_ICU_DATA_FILE}" DESTINATION "${INSTALL_BUNDLE_DATA_DIR}"
  COMPONENT Runtime)

install(FILES "${FLUTTER_LIBRARY}" DESTINATION "${INSTALL_BUNDLE_LIB_DIR}"
  COMPONENT Runtime)

if(PLUGIN_BUNDLED_LIBRARIES)
  install(FILES "${PLUGIN_BUNDLED_LIBRARIES}"
    DESTINATION "${INSTALL_BUNDLE_LIB_DIR}"
    COMPONENT Runtime)
endif()

# Copy the native assets provided by the build.dart from all packages.
set(NATIVE_ASSETS_DIR "${PROJECT_BUILD_DIR}native_assets/windows/")
install(DIRECTORY "${NATIVE_ASSETS_DIR}"
   DESTINATION "${INSTALL_BUNDLE_LIB_DIR}"
   COMPONENT Runtime)

# Fully re-copy the assets directory on each build to avoid having stale files
# from a previous install.
set(FLUTTER_ASSET_DIR_NAME "flutter_assets")
install(CODE "
  file(REMOVE_RECURSE \"${INSTALL_BUNDLE_DATA_DIR}/${FLUTTER_ASSET_DIR_NAME}\")
  " COMPONENT Runtime)
install(DIRECTORY "${PROJECT_BUILD_DIR}/${FLUTTER_ASSET_DIR_NAME}"
  DESTINATION "${INSTALL_BUNDLE_DATA_DIR}" COMPONENT Runtime)

# Install the AOT library on non-Debug builds only.
install(FILES "${AOT_LIBRARY}" DESTINATION "${INSTALL_BUNDLE_DATA_DIR}"
  CONFIGURATIONS Profile;Release
  COMPONENT Runtime)

# Bundle the standalone ONNX Runtime Worker beside the Windows application.
set(SCANNER_DEFAULT_WORKER_RUNTIME_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../artifacts/worker/bixolon-worker")
set(SCANNER_WORKER_RUNTIME_DIR "${SCANNER_DEFAULT_WORKER_RUNTIME_DIR}"
  CACHE PATH "Standalone BIXOLON Worker runtime")
if(EXISTS "${SCANNER_WORKER_RUNTIME_DIR}/bixolon-worker.exe")
  install(DIRECTORY "${SCANNER_WORKER_RUNTIME_DIR}/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/worker"
    COMPONENT Runtime)
else()
  if(SCANNER_RELEASE_BUILD)
    message(FATAL_ERROR
      "Release requires standalone Worker: ${SCANNER_WORKER_RUNTIME_DIR}")
  else()
    message(WARNING
      "Standalone BIXOLON Worker was not found at ${SCANNER_WORKER_RUNTIME_DIR}")
  endif()
endif()

# Bundle the promoted model package beside the Windows application.
set(SCANNER_DEFAULT_MODEL_PACKAGE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../artifacts/packages/bread-worker-1.0.0")
set(SCANNER_LEGACY_MODEL_PACKAGE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../artifacts/packages/bread-worker-0.1.1")
if(NOT SCANNER_MODEL_PACKAGE_DIR OR
   SCANNER_MODEL_PACKAGE_DIR STREQUAL SCANNER_LEGACY_MODEL_PACKAGE_DIR)
  set(SCANNER_MODEL_PACKAGE_DIR "${SCANNER_DEFAULT_MODEL_PACKAGE_DIR}"
    CACHE PATH "Promoted BIXOLON Worker model package" FORCE)
else()
  set(SCANNER_MODEL_PACKAGE_DIR "${SCANNER_MODEL_PACKAGE_DIR}"
    CACHE PATH "Promoted BIXOLON Worker model package")
endif()
if(EXISTS "${SCANNER_MODEL_PACKAGE_DIR}/metadata.json")
  install(DIRECTORY "${SCANNER_MODEL_PACKAGE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/worker/model-package"
    COMPONENT Runtime)
else()
  if(SCANNER_RELEASE_BUILD)
    message(FATAL_ERROR
      "Release requires the composed model package: ${SCANNER_MODEL_PACKAGE_DIR}")
  else()
    message(WARNING
      "BIXOLON Worker model package was not found at ${SCANNER_MODEL_PACKAGE_DIR}")
  endif()
endif()

# Optionally make the Release bundle self-contained for ONNX Runtime CUDA.
# The source directory must contain only redistributable CUDA/cuDNN runtime
# libraries; PyTorch itself is never installed or imported by the Worker.
set(SCANNER_CUDA_RUNTIME_DIR "" CACHE PATH
  "Directory containing redistributable CUDA 13 and cuDNN 9 DLLs")
if(NOT SCANNER_CUDA_RUNTIME_DIR AND DEFINED ENV{BIXOLON_CUDA_DLL_DIR})
  file(TO_CMAKE_PATH "$ENV{BIXOLON_CUDA_DLL_DIR}" SCANNER_CUDA_RUNTIME_DIR)
endif()

set(SCANNER_CUDA_RUNTIME_FILES
  cublas64_13.dll
  cublasLt64_13.dll
  cudart64_13.dll
  cudnn64_9.dll
  cudnn_adv64_9.dll
  cudnn_cnn64_9.dll
  cudnn_engines_precompiled64_9.dll
  cudnn_engines_runtime_compiled64_9.dll
  cudnn_graph64_9.dll
  cudnn_heuristic64_9.dll
  cudnn_ops64_9.dll
  cufft64_12.dll
  nvJitLink_130_0.dll
  nvrtc64_130_0.dll
  nvrtc-builtins64_130.dll
  zlibwapi.dll
)

if(SCANNER_CUDA_RUNTIME_DIR)
  set(SCANNER_CUDA_RUNTIME_PATHS)
  set(SCANNER_CUDA_RUNTIME_COMPLETE TRUE)
  foreach(RUNTIME_FILE IN LISTS SCANNER_CUDA_RUNTIME_FILES)
    set(RUNTIME_PATH "${SCANNER_CUDA_RUNTIME_DIR}/${RUNTIME_FILE}")
    if(EXISTS "${RUNTIME_PATH}")
      list(APPEND SCANNER_CUDA_RUNTIME_PATHS "${RUNTIME_PATH}")
    else()
      set(SCANNER_CUDA_RUNTIME_COMPLETE FALSE)
      if(SCANNER_RELEASE_BUILD)
        message(FATAL_ERROR "CUDA runtime file is missing: ${RUNTIME_PATH}")
      else()
        message(WARNING "CUDA runtime file is missing: ${RUNTIME_PATH}")
      endif()
    endif()
  endforeach()
  if(SCANNER_CUDA_RUNTIME_COMPLETE)
    install(FILES ${SCANNER_CUDA_RUNTIME_PATHS}
      DESTINATION "${CMAKE_INSTALL_PREFIX}/worker/cuda-runtime"
      COMPONENT Runtime)
    message(STATUS
      "Bundling ONNX Runtime CUDA dependencies from "
      "${SCANNER_CUDA_RUNTIME_DIR}")
  else()
    if(SCANNER_RELEASE_BUILD)
      message(FATAL_ERROR "CUDA runtime bundle is incomplete")
    else()
      message(WARNING "CUDA runtime bundle is incomplete")
    endif()
  endif()
else()
  if(SCANNER_RELEASE_BUILD)
    message(FATAL_ERROR "Release requires SCANNER_CUDA_RUNTIME_DIR")
  else()
    message(STATUS "SCANNER_CUDA_RUNTIME_DIR is empty")
  endif()
endif()
