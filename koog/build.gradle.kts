import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidKotlinMultiplatformLibrary)
    alias(libs.plugins.publish)
}

// Koog integration for llama-cpp-kmp: a koog LLMClient backed by llama-cpp-kmp's Inference,
// with text-prompt tool-calling for GGUF models that lack native function calling.
kotlin {
    explicitApi()

    android {
        namespace = "com.kmpile.llama.koog"
        compileSdk = 36
        minSdk = 24
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    jvm()

    @OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    iosX64()
    iosArm64()
    iosSimulatorArm64()
    // macosArm64: koog 1.0.0 (Maven Central) has no macosArm64 variant yet (pending
    // the koog PR). :core supports it; add it here once koog publishes a macos build.

    sourceSets {
        commonMain.dependencies {
            // Only the native-backed core + official koog; everything else (coroutines,
            // serialization, …) comes transitively from koog.
            api(project(":core"))
            api(libs.koog.agents)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

// Optional file-based Maven repository (e.g. a local artifact-repo checkout), same
// as :core. -PartifactRepoUrl=file:///path enables publishAllPublicationsToArtifactRepoRepository.
publishing {
    repositories {
        providers.gradleProperty("artifactRepoUrl").orNull?.let { repoUrl ->
            maven {
                name = "artifactRepo"
                url = uri(repoUrl)
            }
        }
    }
}
