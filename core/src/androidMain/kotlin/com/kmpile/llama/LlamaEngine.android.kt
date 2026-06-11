package com.kmpile.llama

import kotlinx.coroutines.CoroutineDispatcher

internal actual suspend fun platformCreateInference(
    modelSettings: ModelSettings,
    samplingSettings: SamplingSettings,
    progressCallback: (Float) -> Boolean,
    dispatcher: CoroutineDispatcher,
): LlamaEngine {
    System.loadLibrary("llamakmp-jni")
    val engine = JniLlamaEngine(dispatcher)
    if (!engine.load(modelSettings, progressCallback)) {
        engine.close()
        error("Failed to load model: ${modelSettings.modelPath}")
    }
    engine.setSamplingParams(samplingSettings)
    return engine
}

internal actual fun platformMetadata(path: String): ModelMetadata {
    System.loadLibrary("llamakmp-jni")
    return getModelMetadata(path)
}
