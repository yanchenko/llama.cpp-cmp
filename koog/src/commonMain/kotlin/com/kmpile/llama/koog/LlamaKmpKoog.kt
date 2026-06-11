package com.kmpile.llama.koog

import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import com.kmpile.llama.GenerationEvent
import com.kmpile.llama.LlamaEngine
import kotlinx.coroutines.flow.collect

public data object LlamaKmpLLMProvider : LLMProvider("llama.cpp-kmp", "llama.cpp-kmp")

public fun llamaKmpModel(id: String, contextLength: Long = 4_096L): LLModel =
    LLModel(
        provider = LlamaKmpLLMProvider,
        id = id,
        capabilities = listOf(LLMCapability.Completion, LLMCapability.Tools),
        contextLength = contextLength,
    )

/**
 * Wraps **this** engine as a Koog [LLMClient] that **owns** it — closing the client closes the
 * engine. Use for a create-and-use lifecycle. To share/reuse one engine across many clients (so
 * the model isn't reloaded each time), let the caller own it and use [llamaKmpKoogClient] instead.
 */
public fun LlamaEngine.asKoogClient(
    provider: LLMProvider = LlamaKmpLLMProvider,
    maxTokens: Int = 1_024,
): LLMClient = OnDeviceTextLLMClient(
    provider = provider,
    generate = { prompt -> generateText(prompt, maxTokens) },
    onClose = { close() },
)

/**
 * A Koog [LLMClient] whose [LlamaEngine] is supplied lazily by [engine] at generation time. Because
 * Koog calls [engine] inside its `suspend` generate, the *consumer* (e.g. a synchronous factory) never
 * needs `runBlocking` to create the engine. [engine] is expected to return a cached/reused engine; this
 * client does not close it — the caller owns the engine's lifecycle.
 */
public fun llamaKmpKoogClient(
    engine: suspend () -> LlamaEngine,
    provider: LLMProvider = LlamaKmpLLMProvider,
    maxTokens: Int = 1_024,
): LLMClient = OnDeviceTextLLMClient(
    provider = provider,
    generate = { prompt -> engine().generateText(prompt, maxTokens) },
    onClose = {},
)

private suspend fun LlamaEngine.generateText(prompt: String, maxTokens: Int): String = buildString {
    chat(prompt, maxTokens).collect { event ->
        when (event) {
            is GenerationEvent.Token -> append(event.text)
            is GenerationEvent.Error -> error("llama-cpp-kmp generation failed: ${event.error}")
            GenerationEvent.Done, GenerationEvent.Loading -> {}
        }
    }
}
