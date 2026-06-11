#include <jni.h>
#include "llamakmp.h"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cstdlib>

#ifdef __ANDROID__
#include <android/log.h>
#define LCMP_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "LLAMAKMP", __VA_ARGS__)
#define LCMP_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LLAMAKMP", __VA_ARGS__)
#else
#include <cstdio>
#define LCMP_LOGD(...) do { std::fprintf(stderr, "[LLAMAKMP] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define LCMP_LOGE(...) do { std::fprintf(stderr, "[LLAMAKMP] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

struct InferContext {
    JNIEnv*   env;
    jobject   callback;
    jmethodID methodID;
    jlong     inference_ptr = 0;
    // Set once the Java callback throws: a JNI exception is pending, so no further JNI
    // calls may be made from this native frame.
    bool      aborted = false;
};

extern "C"
JNIEXPORT jlong JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_init(JNIEnv *env, jclass clazz) {
    LCMP_LOGD("System properties: %s", llama_print_system_info());
    return init();
}

bool model_load_progress(float progress, void* user_data) {
    auto* ctx = static_cast<InferContext*>(user_data);
    if (ctx->aborted) {
        return false;
    }
    // progress is the 0..1 fraction reported by llama.cpp; returning false aborts the load.
    jboolean keep_going = ctx->env->CallBooleanMethod(ctx->callback, ctx->methodID, progress);
    if (ctx->env->ExceptionCheck()) {
        // Leave the exception pending for the JVM and abort the load.
        ctx->aborted = true;
        return false;
    }
    return keep_going == JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_loadModel(JNIEnv *env,
                        jclass clazz,
                        jlong inference_ptr,
                        jstring path,
                        jint num_of_gpu,
                        jboolean use_mmap,
                        jboolean use_mlock,
                        jobject callback
                        ) {
    auto path_to_model = env->GetStringUTFChars(path, 0);
    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onProgressCallback = env->GetMethodID(callbackClass, "onProgressCallback", "(F)Z");
    if (onProgressCallback == nullptr) {
        LCMP_LOGE("Method onProgressCallback not found");
        env->ReleaseStringUTFChars(path, path_to_model);
        return false;
    }
    auto* ctx = new InferContext{env, callback, onProgressCallback };
    auto is_model_loaded = load_model(
            inference_ptr,
            path_to_model,
            num_of_gpu,
            use_mmap,
            use_mlock,
            model_load_progress,
            ctx
    );
    env->ReleaseStringUTFChars(path, path_to_model);
    // Freed here, exactly once, after the native load returns — not in the progress
    // callback, which may never see progress >= 1.0 (failed/aborted loads).
    delete ctx;

    return is_model_loaded;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_setSamplingParams(JNIEnv *env,
                                jclass clazz,
                                jlong inference_ptr,
                                jfloat temp,
                                jfloat top_p,
                                jfloat min_p,
                                jint top_k) {
    set_sampling_params(
            inference_ptr,
            temp,
            top_p,
            min_p,
            top_k
    );
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_setContextParams(JNIEnv *env,
                               jclass clazz,
                               jlong inference_ptr,
                               jint context_window,
                               jint batch,
                               jint num_of_threads) {
    if (num_of_threads == 0){
        num_of_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency() - 2));
    }
    return set_context_params(
            inference_ptr,
            context_window,
            batch,
            num_of_threads
    );
}

void generation_callback(const char *message, enum generation_event event, void *user_data) {
    auto* ctx = static_cast<InferContext*>(user_data);
    if (!ctx->aborted) {
        jstring messageStr = ctx->env->NewStringUTF(message);
        jint j_event = static_cast<jint>(event);
        ctx->env->CallVoidMethod(ctx->callback, ctx->methodID, messageStr, j_event);
        ctx->env->DeleteLocalRef(messageStr); // safe even with a pending exception
        if (ctx->env->ExceptionCheck()) {
            // The Java callback threw: leave the exception pending for the JVM, stop
            // making JNI calls from this native frame, and cancel the generation.
            ctx->aborted = true;
            cancel_generation(ctx->inference_ptr);
        }
    }
    if (event != LOADING && event != GENERATING){
        delete ctx;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_completion(JNIEnv *env,
                                                  jclass clazz,
                                                  jlong inference_ptr,
                                                  jstring prompt,
                                                  jint max_generation_count,
                                                  jobject callback) {

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onGenerationMethod = env->GetMethodID(callbackClass, "onGeneration", "(Ljava/lang/String;I)V");
    if (onGenerationMethod == nullptr) {
        LCMP_LOGE("Method onGeneration not found");
        return;
    }

    auto prompt_char = env->GetStringUTFChars(prompt, 0);
    LCMP_LOGE("Prompt: %s", prompt_char);
    LCMP_LOGD("Infering...");
    auto * ctx = new InferContext{env, callback, onGenerationMethod, inference_ptr };
    complete(inference_ptr, prompt_char, max_generation_count, generation_callback, ctx);
    env->ReleaseStringUTFChars(prompt, prompt_char);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_chat(JNIEnv *env,
                                                  jclass clazz,
                                                  jlong inference_ptr,
                                                  jstring prompt,
                                                  jint max_generation_count,
                                                  jobject callback) {

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onGenerationMethod = env->GetMethodID(callbackClass, "onGeneration", "(Ljava/lang/String;I)V");
    if (onGenerationMethod == nullptr) {
        LCMP_LOGE("Method onGeneration not found");
        return;
    }

    auto prompt_char = env->GetStringUTFChars(prompt, 0);
    LCMP_LOGE("Prompt: %s", prompt_char);
    auto * ctx = new InferContext{env, callback, onGenerationMethod, inference_ptr };
    chat(inference_ptr, prompt_char, max_generation_count, generation_callback, ctx);
    env->ReleaseStringUTFChars(prompt, prompt_char);
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_getModelMetadata(JNIEnv *env,
                                            jclass clazz,
                                            jstring path) {

    auto path_to_model = env->GetStringUTFChars(path, 0);
    auto model_details = get_model_details(path_to_model);
    env->ReleaseStringUTFChars(path, path_to_model);

    // Fields may be NULL (missing GGUF keys); substitute "" like the Apple actual's
    // .orEmpty(). They are strdup'd by the native side, so free them once copied.
    jstring name = env->NewStringUTF(model_details.name ? model_details.name : "");
    jstring arch = env->NewStringUTF(model_details.architecture ? model_details.architecture : "");
    jlong  context_length = model_details.context_length ? strtoll(model_details.context_length, nullptr, 10) : 0;
    jint   version = model_details.version;
    free((void *) model_details.name);
    free((void *) model_details.architecture);
    free((void *) model_details.context_length);

    jclass modelDetails = env->FindClass("com/kmpile/llama/ModelMetadata");
    if (!modelDetails) return nullptr;

    jmethodID ctor = env->GetMethodID(modelDetails, "<init>", "(ILjava/lang/String;Ljava/lang/String;J)V");
    if (!ctor) {
        env->DeleteLocalRef(modelDetails);
        return nullptr;
    }

    jobject dataObj = env->NewObject(modelDetails, ctor, version, arch, name, context_length);

    env->DeleteLocalRef(name);
    env->DeleteLocalRef(arch);
    env->DeleteLocalRef(modelDetails);

    return dataObj;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_cancelGeneration(JNIEnv *env,jclass clazz,jlong inference_ptr) {
    cancel_generation(inference_ptr);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_kmpile_llama_LlamaKmpNativeKt_cleanUp(JNIEnv *env, jclass clazz, jlong inference_ptr) {
    clean_up(inference_ptr);
}
