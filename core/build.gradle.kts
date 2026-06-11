import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.io.File

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidKotlinMultiplatformLibrary)
    alias(libs.plugins.publish)
}

kotlin {
    explicitApi()

    // AGP 9 KMP Android library target (replaces androidTarget {} + android {}).
    android {
        namespace = "com.kmpile.llama"
        compileSdk = 36
        minSdk = 24
        // Restore Android host (unit) tests for the androidHostTest source set.
        withHostTestBuilder {}.configure {}
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    jvm()

    @OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64(),
        macosArm64(),
    ).forEach {
        it.binaries.framework {
            baseName = "LlamaKmp"
            isStatic = true
        }
        it.compilations.getByName("main") {
            cinterops {
                val llamakmp by creating{
                    // Our glue headers + llama.cpp public headers (from the submodule).
                    val nativeRoot = "${rootProject.projectDir}/native/cpp"
                    includeDirs(
                        "$nativeRoot/src",
                        "$nativeRoot/llama.cpp/include",
                        "$nativeRoot/llama.cpp/ggml/include",
                    )
                }
            }
        }
    }

    // The custom jniMain dependsOn edges below opt out of the auto-applied default
    // hierarchy, so apply it explicitly to keep iOS/apple/native source sets wired.
    applyDefaultHierarchyTemplate()

    sourceSets {
        // Shared JNI source set for Android + desktop JVM (external decls + JniLlamaEngine).
        val commonMain by getting {
            dependencies {
                api(libs.kotlinx.coroutines.core)
            }
        }
        val jniMain by creating { dependsOn(commonMain) }
        val androidMain by getting { dependsOn(jniMain) }
        val jvmMain by getting {
            dependsOn(jniMain)
            // Desktop native libs built by the desktop-jni CMake below.
            resources.srcDir(layout.buildDirectory.dir("generated/jvmNativeResources"))
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

// ---- Desktop (JVM) JNI native build ----
// Builds libllamakmp-jni for the host OS (llama.cpp linked static) and bundles it
// into jvmMain resources; Inference.jvm.kt extracts + System.load()s it at runtime.
private val hostPlatform: String = System.getProperty("os.name").lowercase().let { os ->
    when {
        os.contains("mac") -> "macos"
        os.contains("nux") -> "linux"
        os.contains("win") -> "windows"
        else -> "unknown"
    }
}
private val desktopCmake: String =
    providers.gradleProperty("cmakePath").orNull
        ?: System.getenv("CMAKE_PATH")
        ?: listOf("/opt/homebrew/bin/cmake", "/usr/local/bin/cmake", "/usr/bin/cmake").firstOrNull { File(it).canExecute() }
        ?: "cmake"
private val desktopJniBuildDir = layout.buildDirectory.dir("desktop-jni").get().asFile
private val jvmNativeOutDir = layout.buildDirectory.dir("generated/jvmNativeResources/native/$hostPlatform").get().asFile
// CI builds each OS's lib on its own runner, drops them under generated resources,
// and bundles them without rebuilding: -PllamakmpPrebuiltNatives=true (or env
// LLAMAKMP_PREBUILT_NATIVES=1).
private val nativesPrebuilt: Boolean =
    providers.gradleProperty("llamakmpPrebuiltNatives").orNull == "true" ||
        System.getenv("LLAMAKMP_PREBUILT_NATIVES") == "1"
// Opt-in NVIDIA CUDA backend for the desktop JNI lib (-PllamakmpCuda=true or env
// LLAMAKMP_CUDA=1). Requires the CUDA Toolkit (nvcc) on the build machine, so it's
// off by default (CI runners have no CUDA) — build it locally on an NVIDIA host.
private val cudaEnabled: Boolean =
    providers.gradleProperty("llamakmpCuda").orNull == "true" ||
        System.getenv("LLAMAKMP_CUDA") == "1"

val configureDesktopJni by tasks.registering(Exec::class) {
    group = "llama-native"
    enabled = !nativesPrebuilt
    doFirst { desktopJniBuildDir.mkdirs() }
    val args = mutableListOf(
        desktopCmake, "-S", rootProject.file("native/desktop").absolutePath,
        "-B", desktopJniBuildDir.absolutePath, "-DCMAKE_BUILD_TYPE=Release", "-Wno-dev",
    )
    if (cudaEnabled) args += "-DGGML_CUDA=ON"
    commandLine(args)
}
val buildDesktopJni by tasks.registering(Exec::class) {
    group = "llama-native"
    enabled = !nativesPrebuilt
    dependsOn(configureDesktopJni)
    commandLine(desktopCmake, "--build", desktopJniBuildDir.absolutePath, "--config", "Release")
}
val copyDesktopJni by tasks.registering(Copy::class) {
    group = "llama-native"
    onlyIf { !nativesPrebuilt }
    dependsOn(buildDesktopJni)
    val patterns = when (hostPlatform) {
        "macos" -> listOf("**/*.dylib")
        "linux" -> listOf("**/*.so", "**/*.so.*")
        "windows" -> listOf("**/*.dll")
        else -> emptyList()
    }
    from(fileTree(desktopJniBuildDir) { include(patterns) })
    eachFile { path = name }
    includeEmptyDirs = false
    into(jvmNativeOutDir)
    doFirst { jvmNativeOutDir.deleteRecursively(); jvmNativeOutDir.mkdirs() }
    doLast {
        val libs = jvmNativeOutDir.listFiles()
            ?.filter { it.isFile && it.name != "native-libs.txt" }?.map { it.name }?.sorted().orEmpty()
        jvmNativeOutDir.resolve("native-libs.txt").writeText(libs.joinToString("\n"))
    }
}
// In the prebuilt path, just (re)generate native-libs.txt for each OS folder CI placed.
val writeNativeLibsManifests by tasks.registering {
    group = "llama-native"
    onlyIf { nativesPrebuilt }
    doLast {
        val root = layout.buildDirectory.dir("generated/jvmNativeResources/native").get().asFile
        root.listFiles()?.filter { it.isDirectory }?.forEach { dir ->
            val libs = dir.listFiles()
                ?.filter { it.isFile && it.name != "native-libs.txt" }?.map { it.name }?.sorted().orEmpty()
            dir.resolve("native-libs.txt").writeText(libs.joinToString("\n"))
        }
    }
}
tasks.matching { it.name == "jvmProcessResources" }.configureEach {
    dependsOn(if (nativesPrebuilt) writeNativeLibsManifests else copyDesktopJni)
}

// The JNI .so is built by :native (AGP 9's KMP plugin can't run CMake). Embed
// those libs into THIS module's AAR jniLibs so the published artifact is
// self-contained, instead of declaring a dependency on the unpublished module.
val embedNativeLibs by tasks.registering(Copy::class) {
    dependsOn(":native:assembleRelease")
    val jniLibsDir = layout.projectDirectory.dir("src/androidMain/jniLibs")
    // Clear stale libs (e.g. after a native lib rename) so only current ones ship.
    doFirst { delete(jniLibsDir) }
    val nativeAar = rootProject.file("native/build/outputs/aar/native-release.aar")
    from(zipTree(nativeAar)) {
        include("jni/**")
        eachFile { path = path.removePrefix("jni/") }
    }
    includeEmptyDirs = false
    into(layout.projectDirectory.dir("src/androidMain/jniLibs"))
}
tasks.matching { it.name == "mergeAndroidMainJniLibFolders" }.configureEach {
    dependsOn(embedNativeLibs)
}
