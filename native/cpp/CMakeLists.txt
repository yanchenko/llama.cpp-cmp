cmake_minimum_required(VERSION 3.22.1)

project(llamakmp C CXX)

option(LLAMAKMP_BUILD_APPLE "Build for Apple targets" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LLAMA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# ARM CPU acceleration (KleidiAI i8mm/dotprod kernels) on arm64 Android only.
# Harmless / not applicable elsewhere.
if (ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "Enable KleidiAI ARM kernels" FORCE)
endif ()

# Android GPU: the OpenCL backend (Adreno-optimized) is the only Android GPU
# path llama.cpp supports well — its Vulkan backend is broken (Adreno) or
# slower than CPU (Mali) on mobile drivers. The Khronos ICD loader is built
# static and absorbed into libggml-opencl.so; at runtime it dlopens the
# vendor's libOpenCL.so and reports zero platforms on devices without one
# (emulators, Pixels), so ggml falls back to the CPU backend. Pass
# -DLLAMAKMP_ANDROID_OPENCL=OFF to build CPU-only. Embedding the .cl kernels
# needs a host Python 3 at build time.
option(LLAMAKMP_ANDROID_OPENCL "Enable the OpenCL GPU backend on Android" ON)
if (ANDROID AND LLAMAKMP_ANDROID_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_HEADERS_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-Headers" CACHE PATH "" FORCE)
    add_subdirectory(OpenCL-ICD-Loader build-opencl-icd EXCLUDE_FROM_ALL)
    # Satisfy ggml's find_package(OpenCL REQUIRED) with the in-tree loader:
    # preset cache vars FindOpenCL would otherwise search the sysroot for.
    # "OpenCL" resolves to the static loader target at link time.
    set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-Headers" CACHE PATH "" FORCE)
    set(OpenCL_LIBRARY OpenCL CACHE STRING "" FORCE)
endif ()

# Build llama.cpp (+ llama-common) straight from the submodule instead of
# vendoring copied sources. Bumping llama.cpp = moving the submodule.
add_subdirectory(${LLAMA_SRC} build-llama)

if (LLAMAKMP_BUILD_APPLE)
    add_library(llamakmp STATIC ${SOURCE_DIR}/llamakmp.cpp ${SOURCE_DIR}/Inference.cpp)
else ()
    add_library(llamakmp SHARED ${SOURCE_DIR}/llamakmp.cpp ${SOURCE_DIR}/Inference.cpp)
endif ()

# Export only the LLAMAKMP_API symbols; keep everything else (incl. our internal
# Inference class) hidden. Scoped to this target so llama/ggml exports are intact.
set_target_properties(llamakmp PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(llamakmp PUBLIC
        ${SOURCE_DIR}
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/ggml/include
)

target_compile_features(llamakmp PRIVATE cxx_std_17)
target_compile_options(llamakmp PRIVATE -O3 -DNDEBUG)

# On the shared (Android) lib, restrict the exported ABI to the public C API via
# a linker version script (hides leaked llama/ggml + our internal symbols).
# iOS builds a static lib whose symbols don't leak to the framework consumer.
if (NOT LLAMAKMP_BUILD_APPLE)
    target_link_options(llamakmp PRIVATE
            "-Wl,--version-script=${SOURCE_DIR}/llamakmp.map")
endif ()

# llama.cpp renamed the common target to llama-common (b9102); support both so
# the link survives llama.cpp version bumps.
if (TARGET llama-common)
    target_link_libraries(llamakmp PUBLIC llama llama-common)
else ()
    target_link_libraries(llamakmp PUBLIC llama common)
endif ()
