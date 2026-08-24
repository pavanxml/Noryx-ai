#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <fstream>
#include <iostream> // TODO: remove me
#include <iterator>
#include <sstream>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s --model ./models/bge-base-en-v1.5-f16.gguf --qwen-model ./models/qwen.gguf --top-k 3 --retrieval-threshold 0.60 --context-file private.md --external-context-file fallback.md --chunk-size 100 --chunk-separator .\n", argv[0]);
    LOG("\n");
}

struct chunk {
    // filename
    std::string filename;
    // original file position
    size_t filepos;
    // original text data
    std::string textdata;
    // tokenized text data
    std::vector<llama_token> tokens;
    // embedding
    std::vector<float> embedding;
    // true for context loaded from --external-context-file
    bool is_external = false;
};

struct retrieval_match {
    int   index = 0;
    float score = 0.0f;
};

// chunk file data to chunks of size >= chunk_size
// chunk_separator is the separator between chunks
static std::vector<chunk> chunk_file(const std::string & filename, int chunk_size, const std::string & chunk_separator) {
    std::vector<chunk> chunks;
    std::ifstream f(filename.c_str(), std::ios::binary);

    if (!f.is_open()) {
        LOG_ERR("could not open file %s\n", filename.c_str());
        return chunks;
    }

    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    if (text.empty()) {
        return chunks;
    }
    if (text.size() >= 4 && text.compare(0, 4, "%PDF") == 0) {
        LOG_WRN("%s appears to be a PDF. Extract text first and pass the extracted text file to --context-file.\n", filename.c_str());
    }

    const std::string separator = chunk_separator.empty() ? "\n" : chunk_separator;
    chunk current_chunk;
    current_chunk.filename = filename;
    current_chunk.filepos = 0;

    size_t cursor = 0;
    size_t filepos = 0;
    while (cursor < text.size()) {
        size_t pos = text.find(separator, cursor);
        const bool found_separator = pos != std::string::npos;
        const size_t end = found_separator ? pos + separator.size() : text.size();
        current_chunk.textdata += text.substr(cursor, end - cursor);

        if ((int) current_chunk.textdata.size() >= chunk_size || !found_separator) {
            current_chunk.filename = filename;
            current_chunk.filepos = filepos;
            chunks.push_back(current_chunk);
            filepos += current_chunk.textdata.size();
            current_chunk = chunk();
        }

        cursor = end;
    }

    return chunks;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_process(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), false);

    // run model
    LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                LOG_ERR("%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        common_embd_normalize(embd, out, n_embd, 2);
    }
}

static std::string build_context_string(
        const std::vector<chunk> & chunks,
        const std::vector<retrieval_match> & matches) {
    std::ostringstream context;
    for (int i = 0; i < (int) matches.size(); ++i) {
        const chunk & c = chunks[matches[i].index];
        context << "[source: " << c.filename << ", filepos: " << c.filepos << ", score: " << matches[i].score << "]\n";
        context << c.textdata << "\n\n";
    }
    return context.str();
}

static std::string build_qwen_prompt(
        const std::string & query,
        const std::string & context,
        const char * source) {
    std::ostringstream prompt;
    prompt << "You are a question-answering assistant. Answer only from the provided " << source << " context.\n";
    prompt << "Do not use prior knowledge. Do not infer facts that are not written in the context.\n";
    prompt << "If the context contains the answer, answer directly from that context.\n";
    prompt << "If the answer is not present in the context, say exactly: I don't know based on the available information.\n\n";
    prompt << "Context:\n" << context;
    prompt << "Question:\n" << query << "\n\n";
    prompt << "Answer:";
    return prompt.str();
}

struct qwen_generation {
    llama_model *   model = NULL;
    llama_context * ctx   = NULL;
    llama_sampler * smpl  = NULL;
    const llama_vocab * vocab = NULL;
};

static std::string trim_response(std::string response) {
    const char * markers[] = { "<|im_end|>", "<|endoftext|>" };
    for (const char * marker : markers) {
        const size_t pos = response.find(marker);
        if (pos != std::string::npos) {
            response.resize(pos);
        }
    }
    while (!response.empty() && (response.back() == ' ' || response.back() == '\n' || response.back() == '\r' || response.back() == '\t')) {
        response.pop_back();
    }
    size_t first = 0;
    while (first < response.size() && (response[first] == ' ' || response[first] == '\n' || response[first] == '\r' || response[first] == '\t')) {
        ++first;
    }
    if (first > 0) {
        response.erase(0, first);
    }
    return response;
}

static std::string apply_qwen_template(llama_model * model, const std::string & prompt) {
    const char * tmpl = llama_model_chat_template(model, NULL);
    llama_chat_message msg = { "user", prompt.c_str() };

    int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, NULL, 0);
    if (n < 0) {
        LOG_WRN("failed to apply Qwen chat template, using raw prompt\n");
        return prompt;
    }

    std::vector<char> formatted(n + 1, 0);
    n = llama_chat_apply_template(tmpl, &msg, 1, true, formatted.data(), (int32_t) formatted.size());
    if (n < 0) {
        LOG_WRN("failed to apply Qwen chat template, using raw prompt\n");
        return prompt;
    }

    return std::string(formatted.data(), n);
}

static bool init_qwen_generation(qwen_generation & qwen, const common_params & params) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.split_mode   = params.split_mode;
    model_params.main_gpu     = params.main_gpu;

    qwen.model = llama_model_load_from_file(params.qwen_model.path.c_str(), model_params);
    if (qwen.model == NULL) {
        LOG_ERR("failed to load Qwen model %s\n", params.qwen_model.path.c_str());
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx      = params.n_ctx > 0 ? params.n_ctx : 4096;
    ctx_params.n_batch    = ctx_params.n_ctx;
    ctx_params.n_ubatch   = params.n_ubatch;
    ctx_params.no_perf    = params.no_perf;
    ctx_params.embeddings = false;

    qwen.ctx = llama_init_from_model(qwen.model, ctx_params);
    if (qwen.ctx == NULL) {
        LOG_ERR("failed to create Qwen context\n");
        llama_model_free(qwen.model);
        qwen.model = NULL;
        return false;
    }

    qwen.vocab = llama_model_get_vocab(qwen.model);
    qwen.smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(qwen.smpl, llama_sampler_init_top_k(params.sampling.top_k));
    llama_sampler_chain_add(qwen.smpl, llama_sampler_init_top_p(params.sampling.top_p, params.sampling.min_keep));
    llama_sampler_chain_add(qwen.smpl, llama_sampler_init_temp(params.sampling.temp));
    llama_sampler_chain_add(qwen.smpl, llama_sampler_init_dist(params.sampling.seed));

    return true;
}

static std::string qwen_generate(qwen_generation & qwen, const common_params & params, const std::string & prompt, std::string * sent_prompt) {
    llama_memory_clear(llama_get_memory(qwen.ctx), true);
    llama_sampler_reset(qwen.smpl);

    const std::string formatted_prompt = apply_qwen_template(qwen.model, prompt);
    if (sent_prompt != NULL) {
        *sent_prompt = formatted_prompt;
    }

    const int n_prompt = -llama_tokenize(qwen.vocab, formatted_prompt.c_str(), formatted_prompt.size(), NULL, 0, true, true);
    if (n_prompt <= 0) {
        LOG_ERR("failed to tokenize Qwen prompt\n");
        return "";
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(qwen.vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        LOG_ERR("failed to tokenize Qwen prompt\n");
        return "";
    }

    const int n_predict = params.n_predict > 0 ? params.n_predict : 512;
    const int n_ctx = llama_n_ctx(qwen.ctx);
    if ((int) prompt_tokens.size() >= n_ctx) {
        LOG_ERR("Qwen prompt has %d tokens, but context size is %d\n", (int) prompt_tokens.size(), n_ctx);
        return "";
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    std::string response;
    int n_decode = 0;

    while (n_decode < n_predict) {
        const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(qwen.ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            LOG_WRN("Qwen context size reached before generation completed\n");
            break;
        }

        if (llama_decode(qwen.ctx, batch) != 0) {
            LOG_ERR("Qwen decode failed\n");
            return response;
        }

        llama_token token = llama_sampler_sample(qwen.smpl, qwen.ctx, -1);
        if (llama_vocab_is_eog(qwen.vocab, token)) {
            break;
        }

        char buf[256];
        const int n = llama_token_to_piece(qwen.vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            LOG_ERR("failed to convert Qwen token to text\n");
            break;
        }

        response.append(buf, n);
        batch = llama_batch_get_one(&token, 1);
        ++n_decode;
    }

    return trim_response(response);
}

static void free_qwen_generation(qwen_generation & qwen) {
    if (qwen.smpl != NULL) {
        llama_sampler_free(qwen.smpl);
    }
    if (qwen.ctx != NULL) {
        llama_free(qwen.ctx);
    }
    if (qwen.model != NULL) {
        llama_model_free(qwen.model);
    }
    qwen = qwen_generation();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_RETRIEVAL, print_usage)) {
        return 1;
    }

    // For BERT models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
    params.embedding = true;

    if (params.chunk_size <= 0) {
        LOG_ERR("chunk_size must be positive\n");
        return 1;
    }
    if (params.context_files.empty()) {
        LOG_ERR("context_files must be specified\n");
        return 1;
    }
    if (params.qwen_model.path.empty()) {
        LOG_ERR("qwen_model must be specified with --qwen-model\n");
        return 1;
    }

    LOG_INF("processing private files:\n");
    for (auto & context_file : params.context_files) {
        LOG_INF("%s\n", context_file.c_str());
    }
    if (!params.external_context_files.empty()) {
        LOG_INF("processing external fallback files:\n");
        for (auto & context_file : params.external_context_files) {
            LOG_INF("%s\n", context_file.c_str());
        }
    }

    std::vector<chunk> chunks;
    for (auto & context_file : params.context_files) {
        std::vector<chunk> file_chunk = chunk_file(context_file, params.chunk_size, params.chunk_separator);
        chunks.insert(chunks.end(), file_chunk.begin(), file_chunk.end());
    }
    const int n_private_chunks = (int) chunks.size();
    for (auto & context_file : params.external_context_files) {
        std::vector<chunk> file_chunk = chunk_file(context_file, params.chunk_size, params.chunk_separator);
        for (auto & c : file_chunk) {
            c.is_external = true;
        }
        chunks.insert(chunks.end(), file_chunk.begin(), file_chunk.end());
    }
    LOG_INF("Number of private chunks: %d\n", n_private_chunks);
    LOG_INF("Number of external chunks: %zu\n", chunks.size() - n_private_chunks);
    if (chunks.empty()) {
        LOG_ERR("no chunks were created from the supplied context files\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        LOG_ERR("%s: pooling type NONE not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and trim
    for (auto & chunk : chunks) {
        auto inp = common_tokenize(ctx, chunk.textdata, true, false);
        if (inp.size() > n_batch) {
            LOG_ERR("%s: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        // add eos if not present
        if (llama_vocab_eos(vocab) >= 0 && (inp.empty() || inp.back() != llama_vocab_eos(vocab))) {
            inp.push_back(llama_vocab_eos(vocab));
        }
        chunk.tokens = inp;
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) chunks.size(); i++) {
            LOG_INF("%s: prompt %d: '%s'\n", __func__, i, chunks[i].textdata.c_str());
            LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, chunks[i].tokens.size());
            for (int j = 0; j < (int) chunks[i].tokens.size(); j++) {
                LOG_INF("%6d -> '%s'\n", chunks[i].tokens[j], common_token_to_piece(ctx, chunks[i].tokens[j]).c_str());
            }
            LOG_INF("\n\n");
        }
    }

    // initialize batch
    const int n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd_out = llama_model_n_embd_out(model);
    std::vector<float> embeddings(n_chunks * n_embd_out, 0);
    float * emb = embeddings.data();

    // break into batches
    unsigned int p = 0; // number of prompts processed already
    unsigned int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_chunks; k++) {
        // clamp to n_batch tokens
        auto & inp = chunks[k].tokens;

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch || s >= llama_n_seq_max(ctx)) {
            float * out = emb + p * n_embd_out;
            batch_process(ctx, batch, out, s, n_embd_out);
            common_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + p * n_embd_out;
    batch_process(ctx, batch, out, s, n_embd_out);

    // save embeddings to chunks
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embedding = std::vector<float>(emb + i * n_embd_out, emb + (i + 1) * n_embd_out);
        // clear tokens as they are no longer needed
        chunks[i].tokens.clear();
    }

    struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);

    qwen_generation qwen;
    if (!init_qwen_generation(qwen, params)) {
        llama_batch_free(query_batch);
        llama_backend_free();
        return 1;
    }

    // start loop, receive query and return top k similar chunks based on cosine similarity
    std::string query;
    while (true) {
        LOG("Enter query: ");
        if (!std::getline(std::cin, query)) {
            break;
        }
        std::vector<int32_t> query_tokens = common_tokenize(ctx, query, true);
        if (llama_vocab_eos(vocab) >= 0 && (query_tokens.empty() || query_tokens.back() != llama_vocab_eos(vocab))) {
            query_tokens.push_back(llama_vocab_eos(vocab));
        }

        batch_add_seq(query_batch, query_tokens, 0);

        std::vector<float> query_emb(n_embd_out, 0);
        batch_process(ctx, query_batch, query_emb.data(), 1, n_embd_out);

        common_batch_clear(query_batch);

        // compute cosine similarities and apply the private-first priority policy
        {
            auto score_chunks = [&](bool external) {
                std::vector<retrieval_match> matches;
                for (int i = 0; i < n_chunks; i++) {
                    if (chunks[i].is_external != external) {
                        continue;
                    }
                    retrieval_match match;
                    match.index = i;
                    match.score = common_embd_similarity_cos(chunks[i].embedding.data(), query_emb.data(), n_embd_out);
                    matches.push_back(match);
                }
                std::sort(matches.begin(), matches.end(), [](const retrieval_match & a, const retrieval_match & b) {
                    return a.score > b.score;
                });
                return matches;
            };

            auto take_top = [&](const std::vector<retrieval_match> & matches) {
                const int n_select = params.sampling.top_k <= 0 ? (int) matches.size() : std::min(params.sampling.top_k, (int) matches.size());
                return std::vector<retrieval_match>(matches.begin(), matches.begin() + n_select);
            };

            auto print_matches = [&](const char * label, const std::vector<retrieval_match> & matches) {
                LOG("%s top retrieved chunks:\n", label);
                const int n_print = params.sampling.top_k <= 0 ? (int) matches.size() : std::min(params.sampling.top_k, (int) matches.size());
                for (int i = 0; i < n_print; i++) {
                    const chunk & c = chunks[matches[i].index];
                    LOG("rank: %d\n", i + 1);
                    LOG("filename: %s\n", c.filename.c_str());
                    LOG("filepos: %lld\n", (long long int) c.filepos);
                    LOG("similarity: %f\n", matches[i].score);
                    LOG("textdata:\n%s\n", c.textdata.c_str());
                    LOG("--------------------\n");
                }
                if (n_print == 0) {
                    LOG("none\n");
                }
            };

            std::vector<retrieval_match> private_matches = score_chunks(false);
            std::vector<retrieval_match> external_matches;
            std::vector<retrieval_match> selected;
            const char * selected_source = NULL;

            print_matches("Private context", private_matches);
            if (!private_matches.empty() && private_matches.front().score >= params.retrieval_similarity_threshold) {
                selected_source = "private document";
                selected = take_top(private_matches);
            } else {
                if (!private_matches.empty()) {
                    LOG("Private context best similarity %f is below threshold %f; checking external fallback.\n",
                            private_matches.front().score, params.retrieval_similarity_threshold);
                }
                external_matches = score_chunks(true);
                print_matches("External fallback", external_matches);
                if (!external_matches.empty() && external_matches.front().score >= params.retrieval_similarity_threshold) {
                    selected_source = "external fallback";
                    selected = take_top(external_matches);
                }
            }

            LOG("Selected source: %s\n", selected_source == NULL ? "none" : selected_source);
            if (selected_source != NULL) {
                std::string context = build_context_string(chunks, selected);
                std::string prompt = build_qwen_prompt(query, context, selected_source);
                std::string sent_prompt;
                std::string response = qwen_generate(qwen, params, prompt, &sent_prompt);
                LOG("Final answer:\n%s\n", response.empty() ? "I don't know based on the available information." : response.c_str());
            } else {
                LOG("Final answer:\nI don't know based on the available information.\n");
            }
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    // clean up
    free_qwen_generation(qwen);
    llama_batch_free(query_batch);
    llama_backend_free();
}
