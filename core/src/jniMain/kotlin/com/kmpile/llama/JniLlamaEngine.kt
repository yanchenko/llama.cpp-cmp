package com.kmpile.llama

import kotlin.concurrent.Volatile
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext

internal class JniLlamaEngine(private val dispatcher: CoroutineDispatcher) : LlamaEngine {

    private val inferPtr = init()

    @Volatile
    private var closed = false

    internal suspend fun load(
        modelSettings: ModelSettings,
        progressCallback: (Float) -> Boolean,
    ): Boolean = withContext(dispatcher) {
        val callback = object : Callback {
            override fun onGeneration(text: String, event: Int) {}
            override fun onProgressCallback(progress: Float): Boolean = progressCallback(progress)
        }
        val loaded = loadModel(
            inferencePtr = inferPtr,
            path = modelSettings.modelPath,
            numberOfGpuLayers = modelSettings.numberOfGpuLayers,
            useMmap = modelSettings.useMmap,
            useMlock = modelSettings.useMlock,
            callback = callback,
        )
        loaded && setContextParams(
            inferencePtr = inferPtr,
            contextSize = modelSettings.contextSize,
            batchSize = modelSettings.batchSize,
            numberOfThreads = modelSettings.numberOfThreads,
        )
    }

    override fun setSamplingParams(samplingSettings: SamplingSettings) {
        setSamplingParams(
            inferencePtr = inferPtr,
            temperature = samplingSettings.temperature,
            topP = samplingSettings.topP,
            minP = samplingSettings.minP,
            topK = samplingSettings.topK,
        )
    }

    override fun completion(prompt: String, maxTokens: Int): Flow<GenerationEvent> =
        llamaEventFlow(
            dispatcher = dispatcher,
            start = { onEvent -> completion(inferencePtr = inferPtr, prompt = prompt, maxGenerationCount = maxTokens, callback = eventCallback(onEvent)) },
            cancel = { cancelGeneration(inferPtr) },
        )

    override fun chat(prompt: String, maxTokens: Int): Flow<GenerationEvent> =
        llamaEventFlow(
            dispatcher = dispatcher,
            start = { onEvent -> chat(inferencePtr = inferPtr, prompt = prompt, maxGenerationCount = maxTokens, callback = eventCallback(onEvent)) },
            cancel = { cancelGeneration(inferPtr) },
        )

    override fun cancelGeneration() {
        cancelGeneration(inferPtr)
    }

    override fun close() {
        if (closed) return
        closed = true
        cleanUp(inferPtr)
    }

    private fun eventCallback(onEvent: (GenerationEvent) -> Unit): Callback = object : Callback {
        override fun onGeneration(text: String, event: Int) {
            onEvent(
                when (event) {
                    NativeEvent.LOADING -> GenerationEvent.Loading
                    NativeEvent.GENERATING -> GenerationEvent.Token(text)
                    NativeEvent.TOKENIZE_ERROR -> GenerationEvent.Error(GenerationError.Tokenize)
                    NativeEvent.CONTEXT_EXCEEDED_ERROR -> GenerationEvent.Error(GenerationError.ContextExceeded)
                    NativeEvent.DECODE_ERROR -> GenerationEvent.Error(GenerationError.Decode)
                    NativeEvent.END_OF_GENERATION -> GenerationEvent.Done
                    else -> GenerationEvent.Error(GenerationError.Unknown)
                },
            )
        }

        override fun onProgressCallback(progress: Float): Boolean = true
    }
}

private object NativeEvent {
    const val LOADING = 0
    const val GENERATING = 1
    const val TOKENIZE_ERROR = 2
    const val CONTEXT_EXCEEDED_ERROR = 3
    const val DECODE_ERROR = 4
    const val END_OF_GENERATION = 5
}
