plugins {
    alias(libs.plugins.androidLibrary)
}

// Plain Android library that owns the llama.cpp / JNI native build.
// AGP 9's KMP library plugin no longer supports externalNativeBuild, so the
// CMake build is isolated here and consumed by :core's androidMain.
// Absolute path to the shared native sources (native/cpp) owned by this module.
// Passed to CMake so the entry-point CMakeLists doesn't depend on fragile paths.
val nativeDir = projectDir.resolve("cpp").path.replace("\\", "/")

android {
    namespace = "com.kmpile.llama.nativelib"
    compileSdk = 36
    // Pin NDK >= 28 so native .so are 16 KB page-size aligned by default
    // (Google Play requirement since Nov 2025). NDK 27 would need a manual
    // -Wl,-z,max-page-size=16384 linker flag.
    ndkVersion = "28.2.13676358"

    defaultConfig {
        minSdk = 24

        externalNativeBuild {
            cmake {
                arguments(
                    "-DLLAMAKMP_NATIVE_DIR=$nativeDir",
                    "-DLLAMA_BUILD_COMMON=ON",
                    "-DLLAMA_BUILD_APP=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DLLAMA_OPENSSL=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_NATIVE=OFF",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
            }
        }
        ndk {
            abiFilters.add("arm64-v8a")
            // x86_64 so the library runs on standard Android emulators.
            abiFilters.add("x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}
