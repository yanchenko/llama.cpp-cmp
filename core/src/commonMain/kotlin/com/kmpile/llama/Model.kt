package com.kmpile.llama

public data class ModelSettings(
    public val modelPath: String,
    public val contextSize: Int = 4096,
    public val batchSize: Int = 4096,
    public val numberOfGpuLayers: Int = 0,
    public val numberOfThreads: Int = -1,
    public val useMmap: Boolean = true,
    public val useMlock: Boolean = false,
)

public data class SamplingSettings(
    public val temperature: Float = 0.8f,
    public val topK: Int = 40,
    public val topP: Float = 0.95f,
    public val minP: Float = 0.05f,
)

public data class ModelMetadata(
    public val version: Int,
    public val architecture: String,
    public val name: String,
    public val contextLength: Long,
)

public sealed interface GenerationEvent {

    public data object Loading : GenerationEvent

    public data class Token(public val text: String) : GenerationEvent

    public data class Error(public val error: GenerationError) : GenerationEvent

    public data object Done : GenerationEvent
}

public enum class GenerationError {
    Tokenize,
    ContextExceeded,
    Decode,
    Unknown,
}
