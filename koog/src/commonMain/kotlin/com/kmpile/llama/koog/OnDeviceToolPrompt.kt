package com.kmpile.llama.koog

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

internal data class ToolCall(val name: String, val arguments: String)

internal object OnDeviceToolPrompt {

    fun build(prompt: Prompt, tools: List<ToolDescriptor>): String = buildString {
        appendLine("You route the user's latest message to at most one tool. Available tools:")
        if (tools.isEmpty()) appendLine("  (none)")
        tools.forEach { t ->
            val params = (t.requiredParameters + t.optionalParameters).joinToString(", ") { it.name }
            appendLine("  - ${t.name}: ${t.description}${if (params.isNotBlank()) " [args: $params]" else ""}")
        }
        appendLine()
        appendLine(
            "Reply with ONLY a JSON object — no prose, no markdown. To call a tool: " +
                "{\"tool\":\"<name>\",\"arguments\":{...}}. If no tool fits: {\"tool\":null}.",
        )
        appendLine(
            "Use only the tool names listed above and only their declared arguments. Do not invent a " +
                "tool, an argument, or a value. Use any tool results already shown below instead of " +
                "repeating a call that has already returned.",
        )
        appendLine()
        appendLine("Conversation:")
        prompt.messages.forEach { m ->
            val role = when (m.role) {
                Message.Role.System -> "System"
                Message.Role.User -> "User"
                Message.Role.Assistant -> "Assistant"
            }

            m.parts.forEach { part ->
                when (part) {
                    is MessagePart.Text ->
                        part.text.trim().takeIf { it.isNotEmpty() }?.let { appendLine("$role: $it") }
                    is MessagePart.Tool.Call ->
                        appendLine("$role called ${part.tool} with arguments ${part.args}")
                    is MessagePart.Tool.Result ->
                        appendLine("Tool ${part.tool} returned: ${part.output.trim()}")
                    else -> {}
                }
            }
        }
    }

    fun parse(raw: String, tools: List<ToolDescriptor>, json: Json): ToolCall? {
        val jsonText = extractFirstJsonObject(raw) ?: return null
        val obj = runCatching { json.parseToJsonElement(jsonText).jsonObject }.getOrNull() ?: return null
        val name = obj["tool"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() } ?: return null
        if (tools.none { it.name == name }) return null
        val rawArgs = (obj["arguments"] as? JsonObject) ?: JsonObject(emptyMap())

        val args = JsonObject(
            rawArgs.mapValues { (_, v) ->
                if (v is JsonObject || v is JsonArray) JsonPrimitive(v.toString()) else v
            },
        )
        return ToolCall(name, args.toString())
    }

    private fun extractFirstJsonObject(raw: String): String? {
        val start = raw.indexOf('{')
        if (start < 0) return null
        var depth = 0
        var inString = false
        var escaped = false
        for (i in start until raw.length) {
            val c = raw[i]
            if (inString) {
                when {
                    escaped -> escaped = false
                    c == '\\' -> escaped = true
                    c == '"' -> inString = false
                }
            } else {
                when (c) {
                    '"' -> inString = true
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) return raw.substring(start, i + 1)
                    }
                }
            }
        }
        return null
    }
}
