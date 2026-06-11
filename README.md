# llama.cpp-kmp

A [llama.cpp](https://github.com/ggml-org/llama.cpp) binding for Kotlin Multiplatform. Run GGUF models on-device on Android, iOS, and JVM/desktop.

## Install

```kotlin
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation("com.kmpile:llama:<version>")
        }
    }
}
```

On iOS, add **Accelerate.framework** and **Metal.framework** in Xcode.

## Usage

```kotlin
LlamaEngine.create(ModelSettings(modelPath = "london.gguf")).use {
    val cold = it.chat("Look at the stars")
    cold.collect(::play)
}
```

## GPU acceleration

GPU offload is opt-in per model load: set `ModelSettings(numberOfGpuLayers = 99)`
(`0` = CPU-only, the default). Backends are baked in per platform and fall back
to CPU at runtime when no usable GPU/driver is present:

| Platform | Backend | Notes |
|---|---|---|
| macOS (JVM + `macosArm64`) | Metal | On by default, shaders embedded. |
| iOS device | Metal | On by default; simulator is CPU-only. |
| Windows / Linux (JVM) | Vulkan | On by default; resolves the system Vulkan loader lazily, CPU fallback without a driver. |
| Windows / Linux (JVM, NVIDIA) | CUDA | Opt-in: build with `-PllamakmpCuda=true` (needs the CUDA Toolkit). |
| Android | OpenCL | On by default; optimized for Qualcomm Adreno (Snapdragon 8 Gen 3 / Elite class). Devices without an OpenCL driver (most emulators, Google Tensor/Pixel) fall back to CPU. Best with pure `Q4_0` quantization. |
| wasmJs | — | CPU only. |

llama.cpp's other GPU backends (HIP/ROCm, SYCL, Hexagon NPU, WebGPU) are not
built; on Android the Vulkan backend is deliberately not used — current mobile
drivers make it fail (Adreno) or run slower than CPU (Mali).

## Koog (optional)

The [Koog](https://github.com/JetBrains/koog) LLM client backed by this
binding lives in [koog-box](https://github.com/kmpile/koog-box) as
`:client:llama`, which depends on `com.kmpile:llama` from Maven.

## Versioning &amp; releases

Versions track upstream llama.cpp release tags (e.g. `b9592`); binding-only
revisions on the same upstream append a numeric suffix (`b9592-1`), which
Maven and Gradle order correctly: `b9592 < b9592-1 < b9593`.
[sync-llamacpp.yml](.github/workflows/sync-llamacpp.yml) polls upstream daily
and bumps the pinned submodule.

Every push to `main` publishes a SNAPSHOT of the next version (e.g.
`b9592-1-SNAPSHOT`) to the [Central Portal snapshots
repo](https://central.sonatype.com/repository/maven-snapshots/) via
[publish.yml](.github/workflows/publish.yml). Consume it with:

```kotlin
repositories {
    maven("https://central.sonatype.com/repository/maven-snapshots/")
}
configurations.all {
    resolutionStrategy.cacheChangingModulesFor(0, "seconds")
}
```

Immutable releases to Maven Central are currently disabled — snapshots are
the only published artifacts (see the header comment in publish.yml for how
to re-enable releases).

## License

MIT — see [LICENSE.md](LICENSE.md).
