#include <functional>

#include <string>
#include <vector>

#include "common.h"
#include "Inference.h"
#include "llamakmp.h"

extern "C" int64_t init() {
    auto inference = new Inference();
    return reinterpret_cast<int64_t>(inference);
}

extern "C" bool load_model(int64_t inference_ptr,
                           const char* model_path,
                           int number_of_gpu_layers,
                           bool use_mmap,
                           bool use_mlock,
                           progress_callback callback,
                           void *user_data) {
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    model_settings settings{
        model_path,
        number_of_gpu_layers,
        use_mmap,
        use_mlock,
        callback,
        user_data
    };
    return inference->load_model(settings);
}

extern "C" bool set_context_params(int64_t inference_ptr, int context_size, int batch_size, int number_of_threads) {
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    return inference->init_context(context_size, batch_size, number_of_threads);
}

extern "C" void set_sampling_params(int64_t inference_ptr,
                                    float temperature,
                                    float top_p,
                                    float min_p,
                                    int32_t top_k) {
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    sampling_settings settings{
        temperature,
        top_p,
        min_p,
        top_k,
    };
    inference->set_sampling_params(settings);
}

extern "C" void clean_up(const int64_t inference_ptr) {
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    delete inference;
}

extern "C" void complete(int64_t inference_ptr, const char *prompt, int max_generation_count, GenerationCCallback callback, void* user_data) {
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    std::vector<int32_t> tokens;
    try {
        tokens = inference->initialize_batch(prompt);
    } catch (...) {
        // Never let a C++ exception unwind through the extern "C" boundary; report it
        // as the single terminal event instead (completion() guards its own body).
        if (callback) callback("", TOKENIZE_ERROR, user_data);
        return;
    }
    inference->completion(tokens, max_generation_count, callback, user_data);
}

extern "C" void cancel_generation(int64_t inference_ptr){
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    inference->cancel();
}

extern "C" void chat(int64_t inference_ptr, const char *prompt, int max_generation_count, GenerationCCallback callback, void *user_data) {
    const auto inference = reinterpret_cast<Inference *>(inference_ptr);
    inference->chat(prompt, max_generation_count, callback, user_data);
}

extern "C" model_details get_model_details(const char* model_path) {
    return Inference::get_model_details(model_settings{ model_path });
}
