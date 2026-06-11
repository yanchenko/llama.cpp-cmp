cmake_minimum_required(VERSION 3.19)
project(llamakmp_jni_desktop C CXX)

# Desktop (JVM) JNI library: llama.cpp built static and linked into a single
# libllamakmp-jni.{dylib|so|dll}, loaded from jvmMain resources at runtime.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Don't link Homebrew OpenSSL on macOS (would bake an absolute path into the lib).
if(APPLE)
    set(CMAKE_DISABLE_FIND_PACKAGE_OpenSSL TRUE CACHE BOOL "" FORCE)
endif()

get_filename_component(NATIVE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(CPP_DIR "${NATIVE_ROOT}/cpp")
set(LLAMA_SRC "${CPP_DIR}/llama.cpp")
set(JNI_SRC "${NATIVE_ROOT}/src/main/cpp/llamakmp-jni.cpp")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

# GPU backend: Metal on macOS (ggml auto-enables it), Vulkan on Linux/Windows.
# Vulkan resolves lazily against the system loader at runtime and falls back to
# CPU if there's no driver, so it's safe to bake in. -PllamakmpCuda adds CUDA on
# top (NVIDIA only). Mirrors Llamatik's desktop setup.
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
elseif(NOT DEFINED GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${LLAMA_SRC}" "${CMAKE_BINARY_DIR}/llama-build")

find_package(JNI REQUIRED)

add_library(llamakmp-jni SHARED
    "${CPP_DIR}/src/llamakmp.cpp"
    "${CPP_DIR}/src/Inference.cpp"
    "${JNI_SRC}"
)

target_include_directories(llamakmp-jni PRIVATE
    "${CPP_DIR}/src"
    "${LLAMA_SRC}/include"
    "${LLAMA_SRC}/common"
    "${LLAMA_SRC}/ggml/include"
    ${JNI_INCLUDE_DIRS}
)

target_compile_features(llamakmp-jni PRIVATE cxx_std_17)

# Optimization/NDEBUG come from CMAKE_BUILD_TYPE=Release. common/llama-common
# (renamed in llama.cpp b9102) transitively links llama, so don't list it twice.
if (TARGET llama-common)
    target_link_libraries(llamakmp-jni PRIVATE llama-common)
else ()
    target_link_libraries(llamakmp-jni PRIVATE common)
endif ()

set_target_properties(llamakmp-jni PROPERTIES OUTPUT_NAME "llamakmp-jni")
