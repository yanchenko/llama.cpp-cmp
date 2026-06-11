cmake_minimum_required(VERSION 3.22.1)

project(llamakmp-jni)

# Shared native sources live at <module>/native; LLAMAKMP_NATIVE_DIR is injected
# by build.gradle.kts as an absolute path. Build them into this module's build dir.
add_subdirectory(${LLAMAKMP_NATIVE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/native-build)

add_library(
        llamakmp-jni
        SHARED
        llamakmp-jni.cpp
)
target_link_libraries(llamakmp-jni llamakmp log android)
