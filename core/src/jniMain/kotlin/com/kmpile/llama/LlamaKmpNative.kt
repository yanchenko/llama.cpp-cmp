package com.kmpile.llama

// These must be public top-level declarations so the JNI symbol names stay
// Java_com_kmpile_llama_LlamaKmpNativeKt_*; making them internal would mangle the names and
// break System.loadLibrary binding. They are an implementation detail of the JVM/Android engine —
// consumers use LlamaEngine, not these directly.

public interface Callback {
    public fun onGeneration(text: String, event: Int)

    /** [progress] is a 0..1 fraction; return false to abort the model load. */
    public fun onProgressCallback(progress: Float): Boolean
}

public external fun init(): Long

public external fun loadModel(inferencePtr: Long, path: String, numberOfGpuLayers: Int, useMmap: Boolean, useMlock: Boolean, callback: Callback): Boolean

public external fun setSamplingParams(inferencePtr: Long, temperature: Float, topP: Float, minP: Float, topK: Int)

public external fun setContextParams(inferencePtr: Long, contextSize: Int, batchSize: Int, numberOfThreads: Int): Boolean

public external fun completion(inferencePtr: Long, prompt: String, maxGenerationCount: Int, callback: Callback)

public external fun chat(inferencePtr: Long, prompt: String, maxGenerationCount: Int, callback: Callback)

public external fun getModelMetadata(path: String): ModelMetadata

public external fun cancelGeneration(inferencePtr: Long)

public external fun cleanUp(inferencePtr: Long)
