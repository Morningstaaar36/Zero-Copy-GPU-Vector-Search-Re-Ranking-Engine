# Locates an ONNX Runtime distribution.
#
# Set ONNXRUNTIME_ROOT (cache variable or environment variable) to the root of
# an extracted onnxruntime-linux-x64-gpu release. Defines the imported target
# ONNXRuntime::ONNXRuntime.

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
