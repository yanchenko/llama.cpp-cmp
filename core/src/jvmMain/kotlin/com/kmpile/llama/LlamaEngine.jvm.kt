package com.kmpile.llama

import kotlinx.coroutines.CoroutineDispatcher
import java.io.File
import java.util.Locale

private object JvmNativeLoader {
    @Volatile private var loaded = false

    @Synchronized
    fun load() {
        if (loaded) return
        val os = System.getProperty("os.name").lowercase(Locale.ROOT)
        val platform = when {
            os.contains("mac") -> "macos"
            os.contains("nux") -> "linux"
            os.contains("win") -> "windows"
            else -> error("Unsupported OS for llama-cpp-kmp: $os")
        }
        val resourceRoot = "/native/$platform"

        val names = javaClass.getResourceAsStream("$resourceRoot/native-libs.txt")
            ?.bufferedReader()?.useLines { lines ->
                lines.map(String::trim).filter { it.isNotEmpty() && !it.startsWith("#") }.toList()
            }
            ?: error("llama-cpp-kmp native libraries not bundled for platform: $platform")

        val tempDir = File(System.getProperty("java.io.tmpdir"), "llamakmp-${System.nanoTime()}").apply {
            mkdirs(); deleteOnExit()
        }
        names.forEach { name ->
            val input = javaClass.getResourceAsStream("$resourceRoot/$name")
                ?: error("Missing bundled native library: $resourceRoot/$name")
            val out = File(tempDir, name)
            input.use { i -> out.outputStream().use { o -> i.copyTo(o) } }
            out.deleteOnExit()
            System.load(out.absolutePath)
        }
        loaded = true
    }
}

internal actual suspend fun platformCreateInference(
    modelSettings: ModelSettings,
    samplingSettings: SamplingSettings,
    progressCallback: (Float) -> Boolean,
    dispatcher: CoroutineDispatcher,
): LlamaEngine {
    JvmNativeLoader.load()
    val engine = JniLlamaEngine(dispatcher)
    if (!engine.load(modelSettings, progressCallback)) {
        engine.close()
        error("Failed to load model: ${modelSettings.modelPath}")
    }
    engine.setSamplingParams(samplingSettings)
    return engine
}

internal actual fun platformMetadata(path: String): ModelMetadata {
    JvmNativeLoader.load()
    return getModelMetadata(path)
}
