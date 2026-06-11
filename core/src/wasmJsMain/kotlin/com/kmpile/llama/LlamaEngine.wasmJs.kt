package com.kmpile.llama

import kotlinx.coroutines.CoroutineDispatcher

internal actual suspend fun platformCreateInference(
    modelSettings: ModelSettings,
    samplingSettings: SamplingSettings,
    progressCallback: (Float) -> Boolean,
    dispatcher: CoroutineDispatcher,
): LlamaEngine =
    throw UnsupportedOperationException(
        "llama-cpp-kmp: the wasmJs target has no engine yet (needs an emscripten llama.cpp build).",
    )

internal actual fun platformMetadata(path: String): ModelMetadata =
    throw UnsupportedOperationException(
        "llama-cpp-kmp: the wasmJs target has no engine yet (needs an emscripten llama.cpp build).",
    )
