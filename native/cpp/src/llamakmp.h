#ifndef LLAMAKMP_H
#define LLAMAKMP_H

#include "Inference.h"

#if defined(_WIN32)
#  define LLAMAKMP_API
#else
#  define LLAMAKMP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#include <functional>
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef void (*GenerationCCallback)(const char *message, enum generation_event event, void *user_data);

LLAMAKMP_API int64_t init();

LLAMAKMP_API bool load_model(
        int64_t inference_ptr,
        const char* model_path,
        int number_of_gpu_layers,
        bool use_mmap,
        bool use_mlock,
        progress_callback callback,
        void *user_data);

LLAMAKMP_API bool set_context_params(int64_t inference_ptr, int context_size, int batch_size, int number_of_threads);

LLAMAKMP_API void set_sampling_params(
        int64_t inference_ptr,
        float temperature,
        float top_p,
        float min_p,
        int32_t top_k);

LLAMAKMP_API void complete(
        int64_t inference_ptr,
        const char *prompt,
        int max_generation_count,
        GenerationCCallback callback,
        void *user_data);

LLAMAKMP_API void chat(
        int64_t inference_ptr,
        const char *prompt,
        int max_generation_count,
        GenerationCCallback callback,
        void *user_data);

LLAMAKMP_API void cancel_generation(int64_t inference_ptr);

LLAMAKMP_API void clean_up(int64_t inference_ptr);

LLAMAKMP_API struct model_details get_model_details(const char* model_path);

#ifdef __cplusplus
}
#endif

#endif
