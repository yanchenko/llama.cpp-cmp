package com.kmpile.llama

import llamakmp.clean_up
import llamakmp.generation_event
import llamakmp.get_model_details
import llamakmp.init
import llamakmp.load_model
import llamakmp.set_context_params
import llamakmp.set_sampling_params
import kotlinx.cinterop.ByteVarOf
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.CPointed
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.StableRef
import kotlinx.cinterop.asStableRef
import kotlinx.cinterop.staticCFunction
import kotlinx.cinterop.toKString
import kotlinx.cinterop.useContents
import kotlin.concurrent.Volatile
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

internal actual suspend fun platformCreateInference(
    modelSettings: ModelSettings,
    samplingSettings: SamplingSettings,
    progressCallback: (Float) -> Boolean,
    dispatcher: CoroutineDispatcher,
): LlamaEngine {
    val engine = AppleLlamaEngine(dispatcher)
    if (!engine.load(modelSettings, progressCallback)) {
        engine.close()
        error("Failed to load model: ${modelSettings.modelPath}")
    }
    engine.setSamplingParams(samplingSettings)
    return engine
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun platformMetadata(path: String): ModelMetadata =
    get_model_details(path).useContents {
        val metadata = ModelMetadata(
            version = this.version,
            architecture = this.architecture?.toKString().orEmpty(),
            name = name?.toKString().orEmpty(),
            contextLength = context_length?.toKString()?.toLongOrNull() ?: 0L,
        )
        // The strings are strdup'd by the native side; free them once copied.
        platform.posix.free(this.architecture)
        platform.posix.free(this.name)
        platform.posix.free(this.context_length)
        metadata
    }

@OptIn(ExperimentalForeignApi::class)
internal class AppleLlamaEngine(private val dispatcher: CoroutineDispatcher) : LlamaEngine {

    private val inferencePtr = init()

    @Volatile
    private var closed = false

    internal suspend fun load(
        modelSettings: ModelSettings,
        progressCallback: (Float) -> Boolean,
    ): Boolean = withContext(dispatcher) {
        val onProgressRef = StableRef.create(progressCallback)
        val callback = staticCFunction<Float, CPointer<out CPointed>?, Boolean> { progress, userData ->
            // progress is the 0..1 fraction reported by llama.cpp; false aborts the load.
            val ref = userData?.asStableRef<(Float) -> Boolean>()?.get()
            ref?.invoke(progress) ?: true
        }
        val loaded = load_model(
            inference_ptr = inferencePtr,
            model_path = modelSettings.modelPath,
            number_of_gpu_layers = modelSettings.numberOfGpuLayers,
            use_mmap = modelSettings.useMmap,
            use_mlock = modelSettings.useMlock,
            callback = callback,
            user_data = onProgressRef.asCPointer(),
        )
        onProgressRef.dispose()
        loaded && set_context_params(
            inference_ptr = inferencePtr,
            context_size = modelSettings.contextSize,
            batch_size = modelSettings.batchSize,
            number_of_threads = modelSettings.numberOfThreads,
        )
    }

    override fun setSamplingParams(samplingSettings: SamplingSettings) {
        set_sampling_params(
            inference_ptr = inferencePtr,
            temperature = samplingSettings.temperature,
            top_p = samplingSettings.topP,
            min_p = samplingSettings.minP,
            top_k = samplingSettings.topK,
        )
    }

    override fun completion(prompt: String, maxTokens: Int): Flow<GenerationEvent> =
        llamaEventFlow(
            dispatcher = dispatcher,
            start = { onEvent ->
                generate(onEvent) { ref, cb ->
                    llamakmp.complete(
                        inference_ptr = inferencePtr,
                        prompt = prompt,
                        max_generation_count = maxTokens,
                        callback = cb,
                        user_data = ref.asCPointer(),
                    )
                }
            },
            cancel = { llamakmp.cancel_generation(inferencePtr) },
        )

    override fun chat(prompt: String, maxTokens: Int): Flow<GenerationEvent> =
        llamaEventFlow(
            dispatcher = dispatcher,
            start = { onEvent ->
                generate(onEvent) { ref, cb ->
                    llamakmp.chat(
                        inference_ptr = inferencePtr,
                        prompt = prompt,
                        max_generation_count = maxTokens,
                        callback = cb,
                        user_data = ref.asCPointer(),
                    )
                }
            },
            cancel = { llamakmp.cancel_generation(inferencePtr) },
        )

    override fun cancelGeneration() {
        llamakmp.cancel_generation(inferencePtr)
    }

    override fun close() {
        if (closed) return
        closed = true
        // Abort any in-flight generation, then free on the (single-threaded) dispatcher so
        // the native engine is never deleted while a complete()/chat() call is still running.
        llamakmp.cancel_generation(inferencePtr)
        runBlocking(dispatcher) { clean_up(inferencePtr) }
    }

    // The else IS redundant in platform compiles (the cinterop enum is closed there),
    // but compileAppleMainKotlinMetadata sees generation_event as an expect enum,
    // which can never be exhaustive — without the else, metadata compilation fails.
    @Suppress("REDUNDANT_ELSE_IN_WHEN")
    private fun generate(
        onEvent: (GenerationEvent) -> Unit,
        operation: (StableRef<(GenerationEvent) -> Unit>, CPointer<CFunction<(CPointer<ByteVarOf<Byte>>?, generation_event, CPointer<out CPointed>?) -> Unit>>) -> Unit,
    ) {
        val ref = StableRef.create(onEvent)
        val callback = staticCFunction<CPointer<ByteVarOf<Byte>>?, generation_event, CPointer<out CPointed>?, Unit> { textBytes, event, userData ->
            val text = textBytes?.toKString()
            if (text != null) {
                val onGen = userData?.asStableRef<(GenerationEvent) -> Unit>()?.get()
                when (event) {
                    generation_event.CONTEXT_EXCEEDED_ERROR -> onGen?.invoke(GenerationEvent.Error(GenerationError.ContextExceeded))
                    generation_event.END_OF_GENERATION -> onGen?.invoke(GenerationEvent.Done)
                    generation_event.DECODE_ERROR -> onGen?.invoke(GenerationEvent.Error(GenerationError.Decode))
                    generation_event.TOKENIZE_ERROR -> onGen?.invoke(GenerationEvent.Error(GenerationError.Tokenize))
                    generation_event.GENERATING -> onGen?.invoke(GenerationEvent.Token(text))
                    generation_event.LOADING -> onGen?.invoke(GenerationEvent.Loading)
                    else -> onGen?.invoke(GenerationEvent.Error(GenerationError.Unknown))
                }
            }
        }
        operation(ref, callback)
        ref.dispose()
    }
}
