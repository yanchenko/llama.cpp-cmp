package com.kmpile.llama

import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

public interface LlamaEngine : AutoCloseable {

    public fun setSamplingParams(samplingSettings: SamplingSettings)

    public fun completion(prompt: String, maxTokens: Int = Int.MAX_VALUE): Flow<GenerationEvent>

    public fun chat(prompt: String, maxTokens: Int = Int.MAX_VALUE): Flow<GenerationEvent>

    public fun cancelGeneration()

    override fun close()

    public companion object {
        /**
         * [progressCallback] receives the model-load progress as a 0..1 fraction;
         * return false to abort the load.
         */
        public suspend fun create(
            modelSettings: ModelSettings,
            samplingSettings: SamplingSettings = SamplingSettings(),
            progressCallback: (Float) -> Boolean = { true },
            dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
        ): LlamaEngine =
            platformCreateInference(modelSettings, samplingSettings, progressCallback, dispatcher)

        public fun metadata(path: String): ModelMetadata = platformMetadata(path)
    }
}

// Create the platform engine, load the model, and apply sampling params, returning a ready engine
// (throws if the model fails to load). Loading lives here, not on the public LlamaEngine interface.
internal expect suspend fun platformCreateInference(
    modelSettings: ModelSettings,
    samplingSettings: SamplingSettings,
    progressCallback: (Float) -> Boolean,
    dispatcher: CoroutineDispatcher,
): LlamaEngine

internal expect fun platformMetadata(path: String): ModelMetadata

@OptIn(ExperimentalAtomicApi::class)
internal fun llamaEventFlow(
    dispatcher: CoroutineDispatcher,
    start: ((GenerationEvent) -> Unit) -> Unit,
    cancel: () -> Unit,
): Flow<GenerationEvent> = callbackFlow {
    // Set once generation has finished (terminal event delivered or the native call
    // returned) so awaitClose doesn't fire a stale cancel that could kill a later run.
    val completed = AtomicBoolean(false)
    val job = launch(dispatcher) {
        try {
            start { event ->
                if (event is GenerationEvent.Done || event is GenerationEvent.Error) {
                    completed.store(true)
                }
                trySend(event)
                if (event is GenerationEvent.Done || event is GenerationEvent.Error) {
                    channel.close()
                }
            }
        } finally {
            completed.store(true)
            channel.close()
        }
    }
    awaitClose {
        if (!completed.load()) cancel()
        job.cancel()
    }
}.buffer(Channel.UNLIMITED)
