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
set(SCANNER_BUNDLE_BUILD FALSE)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR FLUTTER_BUILD_MODE STREQUAL "release")
  set(SCANNER_BUNDLE_BUILD TRUE)
endif()

# One version root supplies every component of a distributable build.
get_filename_component(SCANNER_REPOSITORY_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(SCANNER_VERSION_ROOT "" CACHE PATH "Prepared BIXOLON version root")
if(NOT SCANNER_VERSION_ROOT AND DEFINED ENV{SCANNER_VERSION_ROOT})
  file(TO_CMAKE_PATH "$ENV{SCANNER_VERSION_ROOT}" SCANNER_VERSION_ROOT)
endif()
if(NOT SCANNER_VERSION_ROOT)
  set(SCANNER_VERSION_ROOT
    "${SCANNER_REPOSITORY_ROOT}/artifacts/versions/0.0.2")
endif()
set(SCANNER_MODEL_PACKAGE_DIR "${SCANNER_VERSION_ROOT}/staging/runtime"
  CACHE PATH "Versioned BIXOLON Worker model package" FORCE)
set(SCANNER_CATALOG_DIR "${SCANNER_VERSION_ROOT}/staging/catalog"
  CACHE PATH "Versioned BIXOLON Store Catalog" FORCE)
set(SCANNER_WORKER_RUNTIME_DIR
  "${SCANNER_VERSION_ROOT}/worker-build/bixolon-worker"
  CACHE PATH "Versioned standalone BIXOLON Worker runtime" FORCE)
set(SCANNER_CUDA_RUNTIME_DIR "${SCANNER_VERSION_ROOT}/staging/cuda-runtime"
  CACHE PATH "Versioned redistributable CUDA runtime" FORCE)
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

# Fully re-copy the Worker tree so excluded modules from an older PyInstaller
# build cannot survive in the release bundle.
install(CODE "
  file(REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/worker\")
  " COMPONENT Runtime)

# Bundle the standalone ONNX Runtime Worker beside the Windows application.
if(EXISTS "${SCANNER_WORKER_RUNTIME_DIR}/bixolon-worker.exe")
  install(DIRECTORY "${SCANNER_WORKER_RUNTIME_DIR}/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/worker"
    COMPONENT Runtime)
else()
  if(SCANNER_BUNDLE_BUILD)
    message(FATAL_ERROR
      "Version bundle requires standalone Worker: ${SCANNER_WORKER_RUNTIME_DIR}")
  else()
    message(WARNING
      "Standalone BIXOLON Worker was not found at ${SCANNER_WORKER_RUNTIME_DIR}")
  endif()
endif()

# Bundle the versioned model package beside the Windows application.
if(EXISTS "${SCANNER_MODEL_PACKAGE_DIR}/metadata.json")
  install(DIRECTORY "${SCANNER_MODEL_PACKAGE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/worker/model-package"
    COMPONENT Runtime)
else()
  if(SCANNER_BUNDLE_BUILD)
    message(FATAL_ERROR
      "Version bundle requires the model package: ${SCANNER_MODEL_PACKAGE_DIR}")
  else()
    message(WARNING
      "BIXOLON Worker model package was not found at ${SCANNER_MODEL_PACKAGE_DIR}")
  endif()
endif()

# Bundle the checksum-verified Store Catalog. No key or signature is copied.
if(EXISTS "${SCANNER_CATALOG_DIR}/catalog.json")
  install(DIRECTORY "${SCANNER_CATALOG_DIR}/"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/worker/store-catalog"
    COMPONENT Runtime)
elseif(SCANNER_BUNDLE_BUILD)
  message(FATAL_ERROR
    "Version bundle requires the Store Catalog: ${SCANNER_CATALOG_DIR}")
endif()

# Make the version bundle self-contained for ONNX Runtime CUDA.
# The source directory must contain only redistributable CUDA/cuDNN runtime
# libraries; PyTorch itself is never installed or imported by the Worker.
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
      if(SCANNER_BUNDLE_BUILD)
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
    if(SCANNER_BUNDLE_BUILD)
      message(FATAL_ERROR "CUDA runtime bundle is incomplete")
    else()
      message(WARNING "CUDA runtime bundle is incomplete")
    endif()
  endif()
else()
  if(SCANNER_BUNDLE_BUILD)
    message(FATAL_ERROR "Version bundle requires SCANNER_CUDA_RUNTIME_DIR")
  else()
    message(STATUS "SCANNER_CUDA_RUNTIME_DIR is empty")
  endif()
endif()
