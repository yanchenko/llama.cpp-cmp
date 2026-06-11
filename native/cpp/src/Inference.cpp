#include "Inference.h"

#include "gguf.h"
#include "llamakmp.h"
#include <iostream>
#include <cstring>
#include <cstdio>
#include <thread>
#include <exception>

#include "common.h"
#include "chat.h"

Inference::Inference() {
    llama_backend_init();
}

Inference::~Inference() {
    clean_up();
}

bool Inference::load_model(const model_settings &settings) {
    if (model){
        reset();
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = settings.number_of_gpu_layers;
    model_params.use_mmap = settings.use_mmap;
    model_params.use_mlock = settings.use_mlock;
    model_params.progress_callback = settings.callback;
    model_params.progress_callback_user_data = settings.progress_callback_user_data;
    model = llama_model_load_from_file(settings.model_path, model_params);
    if (!model) {
        clean_up();
        return false;
    }
    return true;
}

bool Inference::init_context(int context_length, int n_batch, int number_of_threads) {
    llama_context_params ctx_param = llama_context_default_params();
    ctx_param.n_ctx = context_length;
    ctx_param.n_batch = n_batch;
    ctx_param.n_ubatch = n_batch;
    ctx_param.n_threads = number_of_threads;
    ctx_param.n_threads_batch = number_of_threads;

    ctx = llama_init_from_model(model, ctx_param);
    vocab = llama_model_get_vocab(model);

    return ctx != nullptr && vocab != nullptr;
}

model_details Inference::get_model_details(const model_settings &settings) {
    model_details details{};
    const gguf_init_params params = {
        false,
        nullptr,
   };
    gguf_context * ctx = gguf_init_from_file(settings.model_path, params);
    if (!ctx) return details;
    details.version = (int) gguf_get_version(ctx);

    const int64_t n_kv = gguf_get_n_kv(ctx);

    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);
        const gguf_type type = gguf_get_kv_type(ctx, i);

        if (strcmp(key, "general.architecture") == 0 && type == GGUF_TYPE_STRING) {
            details.architecture = strdup(gguf_get_val_str(ctx, i));
        } else if (strcmp(key, "general.name") == 0 && type == GGUF_TYPE_STRING) {
            details.name = strdup(gguf_get_val_str(ctx, i));
        } else if (strstr(key, "context_length") != nullptr) {
            if (type == GGUF_TYPE_STRING) {
                details.context_length = strdup(gguf_get_val_str(ctx, i));
            } else if (type == GGUF_TYPE_UINT32) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%u", gguf_get_val_u32(ctx, i));
                details.context_length = strdup(buf);
            }
        }
    }
    gguf_free(ctx);
    return details;
}

void Inference::check_context_and_resize(int n_batch) {
    const int n_ctx = llama_n_ctx(ctx);
    const int n_used = n_past;

    if (n_used + n_batch > n_ctx) {
        const int n_discard = (n_used - n_keep) / 2;

        if (n_discard > 0) {
            llama_memory_t mem = llama_get_memory(ctx);
            llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(mem, 0, n_keep + n_discard, n_used, -n_discard);
            n_past -= n_discard;

            printf("Context resized:\n");
            printf("  - Kept: %d tokens\n", n_keep);
            printf("  - Discarded: %d tokens\n", n_discard);
            printf("  - Remaining: %d tokens\n", n_past);
        }
    }
}

void Inference::set_sampling_params(const sampling_settings settings) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    if (smpl) llama_sampler_free(smpl);
    smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(settings.min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(settings.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(settings.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(settings.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

std::vector<int32_t> Inference::initialize_batch(const std::string &prompt) {
    llama_batch_free(batch);
    batch = llama_batch{};
    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        prompt.length(),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    // Tokenization failed (negative) or produced no tokens: return empty so
    // completion() can report TOKENIZE_ERROR; batch stays zeroed (safe to free).
    if (n_tokens <= 0) {
        return {};
    }

    tokens.resize(n_tokens);
    // llama_batch_init allocates with the NULL seq_id sentinel that llama_batch_free expects.
    batch = llama_batch_init(n_tokens, 0, 1);

    return tokens;
}

void Inference::completion(
    const std::vector<int32_t> &tokens,
    size_t max_tokens,
    const InferenceCallback &callback,
    void* user_data) {

    // Once a terminal event (anything but LOADING/GENERATING) has been delivered, the
    // receiver may free its context, so no further callbacks are allowed after it.
    bool terminal_sent = false;
    auto emit = [&](const char *message, generation_event event) {
        if (event != LOADING && event != GENERATING) {
            terminal_sent = true;
            // Reset the cancel flag at the END of a generation (not the start) so a
            // cancel that lands just before the loop starts is honored instead of
            // being wiped out.
            generation_cancelled = false;
        }
        callback(message, event, user_data);
    };
  try {
    if (tokens.empty()) {
        emit("", TOKENIZE_ERROR);
        return;
    }

    // Stateless per call, like chat(): clear prior KV-cache state so a repeated
    // completion() on the same engine starts fresh.
    if (ctx) llama_memory_clear(llama_get_memory(ctx), true);

    common_batch_clear(batch);

    for (auto i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        emit("", DECODE_ERROR);
        return;
    }

    for (size_t i = batch.n_tokens; i <= max_tokens; ++i) {
        if (generation_cancelled){
            emit("", END_OF_GENERATION);
            break;
        }
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id) || i == max_tokens) {
            emit("", END_OF_GENERATION);
            break;
        }

        auto generated = common_token_to_piece(ctx, new_token_id);
        emit(generated.c_str(), GENERATING);

        common_batch_clear(batch);
        common_batch_add(batch, new_token_id, i, { 0 }, true);

        if (llama_decode(ctx, batch)) {
            emit("", DECODE_ERROR);
            break;
        }
    }

    // The loop body may never run (e.g. max_tokens < batch.n_tokens); still deliver
    // exactly one terminal event so the receiver can release its context.
    if (!terminal_sent) {
        emit("", END_OF_GENERATION);
    }
  } catch (const std::exception & e) {
    fprintf(stderr, "completion() failed: %s\n", e.what());
    if (!terminal_sent) emit("", DECODE_ERROR);
  } catch (...) {
    fprintf(stderr, "completion() failed: unknown native error\n");
    if (!terminal_sent) emit("", DECODE_ERROR);
  }
}

void Inference::chat(
    const std::string &prompt,
    size_t max_tokens,
    const InferenceCallback &callback,
    void *user_data) {
    // Once a terminal event (anything but LOADING/GENERATING) has been delivered, the
    // receiver may free its context, so no further callbacks are allowed after it.
    bool terminal_sent = false;
    auto emit = [&](const char *message, generation_event event) {
        if (event != LOADING && event != GENERATING) {
            terminal_sent = true;
            // Reset the cancel flag at the END of a generation (not the start) so a
            // cancel that lands just before the loop starts is honored instead of
            // being wiped out.
            generation_cancelled = false;
        }
        callback(message, event, user_data);
    };
  try {
    emit("", LOADING);

    // Stateless per call: callers (e.g. a Koog client) re-send the whole conversation each
    // time, so clear prior state and process this prompt fresh. This lets one cached engine
    // be reused across turns/requests without stale KV-cache state confusing the next turn.
    chat_history.clear();
    prev_len = 0;
    n_past = 0;
    if (ctx) llama_memory_clear(llama_get_memory(ctx), true);

    chat_history.push_back({"user", prompt});

    // Render the model's own chat template through jinja (minja) so modern templates
    // (gemma4, etc.) that the legacy llama_chat_apply_template can't match still work.
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    auto render = [&](bool add_generation_prompt) {
        common_chat_templates_inputs in;
        in.use_jinja = true;
        in.add_generation_prompt = add_generation_prompt;
        for (const auto & m : chat_history) {
            common_chat_msg msg;
            msg.role = m.first;
            msg.content = m.second;
            in.messages.push_back(msg);
        }
        return common_chat_templates_apply(tmpls.get(), in).prompt;
    };

    const std::string formatted = render(true);
    if ((int) formatted.size() < prev_len) {
        fprintf(stderr, "failed to apply the chat template\n");
        emit("", TOKENIZE_ERROR);
        return;
    }
    const std::string formatted_prompt = formatted.substr(prev_len);

    auto generate = [&](const std::string & prompt) {

        std::string response;

        const bool is_first = (n_past == 0);

        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        if (n_prompt_tokens <= 0) {
            emit("", TOKENIZE_ERROR);
            return response;
        }
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            emit("", TOKENIZE_ERROR);
            return response;
        }

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        int count = 0;
        while (true) {
            if (generation_cancelled){
                emit("", END_OF_GENERATION);
                break;
            }

            check_context_and_resize(batch.n_tokens);

            int n_ctx = llama_n_ctx(ctx);
            int n_ctx_used = n_past;
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                emit("", CONTEXT_EXCEEDED_ERROR);
                break;
            }

            if (llama_decode(ctx, batch)) {
                emit("", DECODE_ERROR);
                break;
            }
            n_past += batch.n_tokens;

            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                emit("", END_OF_GENERATION);
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                emit("", DECODE_ERROR);
                break;
            }
            std::string piece(buf, n);
            emit(piece.c_str(), GENERATING);
            count++;
            response += piece;

            if (count >= max_tokens && max_tokens > 0) {
                emit("", END_OF_GENERATION);
                break;
            }
            generated++;

            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };

    std::string response = generate(formatted_prompt);

    chat_history.push_back({"assistant", response});
    prev_len = (int) render(false).size();
  } catch (const std::exception & e) {
    fprintf(stderr, "chat() failed: %s\n", e.what());
    if (!terminal_sent) emit("", DECODE_ERROR);
  } catch (...) {
    fprintf(stderr, "chat() failed: unknown native error\n");
    if (!terminal_sent) emit("", DECODE_ERROR);
  }
}

void Inference::cancel() {
    generation_cancelled = true;
}

void Inference::clean_up() {
    // Intentionally no llama_backend_free() here: the backend is process-lifetime state
    // shared by every Inference instance, so freeing it per instance would break any
    // other instances that are still alive. Leaving it initialized is safe.
    reset();
}

void Inference::reset() {
    chat_history.clear();
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    if (smpl) llama_sampler_free(smpl);
    llama_batch_free(batch);
    ctx = nullptr;
    model = nullptr;
    smpl = nullptr;
    vocab = nullptr;
    batch = llama_batch{};
    n_past = 0;
    prev_len = 0;
}
