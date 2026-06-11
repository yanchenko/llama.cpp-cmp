#ifndef INFERENCE_H
#define INFERENCE_H

#include "llama.h"

enum generation_event {
    LOADING,
    GENERATING,
    TOKENIZE_ERROR,
    CONTEXT_EXCEEDED_ERROR,
    DECODE_ERROR,
    END_OF_GENERATION
};

struct model_details{
    int version;
    const char* architecture;
    const char* name;
    const char* context_length;
};

typedef bool (*progress_callback)(float progress, void * user_data);

#ifdef __cplusplus
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>

using InferenceCallback = std::function<void(const char*, generation_event, void*)>;

struct model_settings {
    const char* model_path;
    int number_of_gpu_layers = 0;
    bool use_mmap = true;
    bool use_mlock = true;
    progress_callback callback = nullptr;
    void * progress_callback_user_data = nullptr;
    int number_of_threads = -1;
    int context = 512;
    int batch_size = 512;
};

struct sampling_settings {
    float temp = 0.8f;
    float top_p = 0.95f;
    float min_p = 0.05f;
    int32_t top_k = 40;
};

class Inference {
    llama_context* ctx = nullptr;
    llama_model* model = nullptr;
    llama_sampler * smpl = nullptr;
    const llama_vocab *vocab = nullptr;
    llama_batch batch {};
    std::vector<std::pair<std::string, std::string>> chat_history;
    int prev_len = 0;
    // Written from another thread by cancel(); atomic to avoid a data race with the
    // generation loop reading it.
    std::atomic<bool> generation_cancelled{false};
    int generated = 0;
    int n_keep = 0;

    llama_pos n_past = 0;

public:
    Inference();
    ~Inference();

    bool load_model(const model_settings &settings);
    bool init_context(int context_length, int n_batch, int number_of_threads = -1);

    void set_sampling_params(sampling_settings settings);

    std::vector<int32_t> initialize_batch(const std::string& prompt);
    void completion(const std::vector<int32_t>& tokens, size_t max_tokens = 128, const InferenceCallback& callback = {}, void* user_data = nullptr);

    void chat(const std::string& prompt, size_t max_tokens = 128, const InferenceCallback& callback = {}, void* user_data = nullptr);
    void cancel();
    void clean_up();
    void reset();

    static model_details get_model_details(const model_settings &settings);
private:
    void check_context_and_resize(int n_batch);
};

#endif

#endif
