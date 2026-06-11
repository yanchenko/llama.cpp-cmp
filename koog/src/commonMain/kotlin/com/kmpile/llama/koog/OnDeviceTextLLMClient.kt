package com.kmpile.llama.koog

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import ai.koog.prompt.message.ResponseMetaInfo
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.json.Json
import kotlin.uuid.Uuid

internal class OnDeviceTextLLMClient(
    private val provider: LLMProvider,
    private val generate: suspend (String) -> String,
    private val onClose: () -> Unit = {},
    private val json: Json = Json { ignoreUnknownKeys = true; isLenient = true },
) : LLMClient() {

    override fun llmProvider(): LLMProvider = provider

    override fun close() {
        runCatching { onClose() }
    }

    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Message.Assistant {
        val text = try {
            generate(OnDeviceToolPrompt.build(prompt, tools))
        } catch (c: CancellationException) {
            throw c
        } catch (t: Throwable) {
            return Message.Assistant("NONE (on-device-error)", ResponseMetaInfo.Empty)
        }

        val call = OnDeviceToolPrompt.parse(text, tools, json)
            ?: return Message.Assistant(text.trim(), ResponseMetaInfo.Empty)
        return Message.Assistant(
            listOf(MessagePart.Tool.Call(id = Uuid.random().toString(), tool = call.name, args = call.arguments)),
            ResponseMetaInfo.Empty,
        )
    }

    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        throw UnsupportedOperationException("on-device text client does not support moderation")

    override suspend fun embed(text: String, model: LLModel): List<Double> =
        throw UnsupportedOperationException("on-device text client does not support embeddings")

    override suspend fun embed(inputs: List<String>, model: LLModel): List<List<Double>> =
        throw UnsupportedOperationException("on-device text client does not support embeddings")
}
