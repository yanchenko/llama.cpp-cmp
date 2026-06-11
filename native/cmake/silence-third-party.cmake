# llama.cpp/ggml (and the OpenCL ICD loader on Android) compile with their own
# -Wall/-Wextra/-Wpedantic. Their warnings aren't ours to fix, and with the Xcode
# toolchain the Apple SDK lapack.h alone emits 100k+ -Wnullability-extension
# warnings — past GitHub's CI log truncation limit. A global -Wno-* in
# CMAKE_C/CXX_FLAGS can't win: per-target compile options are appended after it,
# so ggml's -Wpedantic re-enables the lot. Silence each third-party target
# instead (-w / /w beats any -W); llamakmp's own sources keep full warnings.
function(llamakmp_silence_third_party_warnings dir)
    get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        # Only compilable types — skips INTERFACE/UTILITY (e.g. ggml-vulkan's
        # custom shader-gen targets), which reject compile options.
        if(type MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
            if(MSVC)
                target_compile_options("${target}" PRIVATE /w)
            else()
                target_compile_options("${target}" PRIVATE -w)
            endif()
        endif()
    endforeach()
    get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
        llamakmp_silence_third_party_warnings("${subdir}")
    endforeach()
endfunction()
