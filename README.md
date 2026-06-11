# llama.cpp-kmp

A [llama.cpp](https://github.com/ggml-org/llama.cpp) binding for Kotlin Multiplatform. Run GGUF models on-device on Android, iOS, and JVM/desktop.

## Install

```kotlin
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation("com.kmpile:llama-cpp-kmp:<version>")
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

Use it as a [Koog](https://github.com/JetBrains/koog) LLM client. Add the integration — it pulls in `llama-cpp-kmp` automatically:

```kotlin
commonMain.dependencies {
    implementation("com.kmpile:llama-cpp-kmp-koog:<version>")
}
```

then wrap a loaded `LlamaEngine`:

```kotlin
val executor = SingleLLMPromptExecutor(inference.asKoogClient())
val model = llamaKmpModel(id = "yellow")
// pass `executor` + `model` to a Koog AIAgent
```

## License

MIT — see [LICENSE.md](LICENSE.md).
