// inference/src/main.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Edge Inference Server
//
// What this does:
//   1. Memory-maps the .onnx file (fast cold start, low RAM)
//   2. Creates an ONNX Runtime inference session
//   3. Preprocesses images using SIMD-accelerated normalization
//   4. Runs inference and returns JSON predictions
//   5. Exposes a minimal HTTP server so Python / mobile / anything can call it
//
// Build:
//   See docker/Dockerfile.inference for the full build command.
//   Locally: cmake -B build && cmake --build build
//
// Dependencies:
//   - ONNX Runtime C++ API  (header-only path + shared lib)
//   - cpp-httplib            (single-header HTTP server)
//   - nlohmann/json          (single-header JSON)
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// mmap for cold-start model loading
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// Single-header HTTP server (vendored in inference/include/)
#include "httplib.h"

// Single-header JSON
#include "json.hpp"

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Config (injected at compile time via -D flags from CMake)
// ─────────────────────────────────────────────────────────────────────────────
#ifndef MODEL_PATH
#define MODEL_PATH "/app/model/student_int8.onnx"
#endif

#ifndef SERVER_PORT
#define SERVER_PORT 8080
#endif

#ifndef INPUT_H
#define INPUT_H 224
#endif

#ifndef INPUT_W
#define INPUT_W 224
#endif

#ifndef INPUT_C
#define INPUT_C 3
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Memory-mapped model loader
// ─────────────────────────────────────────────────────────────────────────────

struct MappedModel {
    void*  addr = nullptr;
    size_t size = 0;
    int    fd   = -1;

    // Load model via mmap — the OS pages in only what's actually accessed,
    // so cold-start time is milliseconds instead of seconds.
    static MappedModel load(const char* path) {
        MappedModel m;
        m.fd = open(path, O_RDONLY);
        if (m.fd < 0) {
            throw std::runtime_error(std::string("Cannot open model: ") + path);
        }

        struct stat st;
        fstat(m.fd, &st);
        m.size = static_cast<size_t>(st.st_size);

        m.addr = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
        if (m.addr == MAP_FAILED) {
            close(m.fd);
            throw std::runtime_error("mmap failed for model file");
        }

        // Advise the kernel: we'll read sequentially (helps prefetcher)
        madvise(m.addr, m.size, MADV_SEQUENTIAL);

        std::cout << "[loader] Model mapped: " << path
                  << "  (" << (m.size / 1024 / 1024) << " MB)\n";
        return m;
    }

    ~MappedModel() {
        if (addr && addr != MAP_FAILED) munmap(addr, size);
        if (fd >= 0) close(fd);
    }

    // Non-copyable (owns the mapping)
    MappedModel(const MappedModel&) = delete;
    MappedModel& operator=(const MappedModel&) = delete;
    MappedModel(MappedModel&&) = default;
    MappedModel() = default;
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD-accelerated image preprocessing
// ─────────────────────────────────────────────────────────────────────────────
// We normalize each pixel channel:
//   output = (pixel / 255.0 - mean) / std
//
// ImageNet normalization constants:
//   mean = [0.485, 0.456, 0.406]
//   std  = [0.229, 0.224, 0.225]
//
// SIMD: on ARM (Raspberry Pi) we use NEON; on x86 we use SSE2.
// The compiler auto-vectorizes this loop when -O3 + target flags are set.
// ─────────────────────────────────────────────────────────────────────────────

static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3]  = {0.229f, 0.224f, 0.225f};

std::vector<float> preprocess(
    const std::vector<uint8_t>& raw_rgb,   // HxWx3 uint8 (interleaved RGB)
    int H, int W
) {
    // ONNX Runtime expects NCHW format (channels first, not HWC)
    // Output layout: [C, H, W] = [3, H, W]
    const int num_pixels = H * W;
    std::vector<float> tensor(3 * num_pixels);

    // Split interleaved HWC → planar CHW, normalize in the same pass.
    // The inner loop is trivially vectorizable by the compiler.
    const float inv_255 = 1.0f / 255.0f;
    for (int i = 0; i < num_pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = static_cast<float>(raw_rgb[i * 3 + c]) * inv_255;
            tensor[c * num_pixels + i] = (v - kMean[c]) / kStd[c];
        }
    }
    return tensor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference engine
// ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine {
public:
    InferenceEngine(const void* model_data, size_t model_size) {
        // ONNX Runtime environment (one per process)
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "edge-cv");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);              // match hardware CPU count
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL // fuses ops, removes dead nodes
        );

        // Enable XNNPACK on ARM for hardware-accelerated INT8 kernels
        // XNNPACK provides optimized NEON SIMD kernels on ARM processors
        OrtSessionOptionsAppendExecutionProvider_XNNPACK(opts, {});

        // Load from memory (the mmap buffer) — no extra copy
        session_ = Ort::Session(env_, model_data, model_size, opts);

        // Cache input/output names
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name  = session_.GetInputNameAllocated(0, alloc);
        auto out_name = session_.GetOutputNameAllocated(0, alloc);
        input_name_  = in_name.get();
        output_name_ = out_name.get();

        std::cout << "[engine] Session created. Input: " << input_name_
                  << "  Output: " << output_name_ << "\n";
    }

    struct Result {
        std::vector<float> scores;   // softmax probabilities
        int   top_class;
        float top_score;
        float latency_ms;
    };

    Result run(const std::vector<float>& input_tensor, int H, int W) {
        auto t0 = Clock::now();

        // Describe the input shape: [batch=1, C, H, W]
        std::array<int64_t, 4> shape = {1, INPUT_C, H, W};

        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(input_tensor.data()),
            input_tensor.size(),
            shape.data(), shape.size()
        );

        const char* in_names[]  = {input_name_.c_str()};
        const char* out_names[] = {output_name_.c_str()};

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            in_names, &in_tensor, 1,
            out_names, 1
        );

        auto t1 = Clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // Collect raw logits
        float* logits = outputs[0].GetTensorMutableData<float>();
        size_t num_classes = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // Softmax: convert logits → probabilities
        std::vector<float> scores(logits, logits + num_classes);
        float max_logit = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (auto& s : scores) { s = std::exp(s - max_logit); sum += s; }
        for (auto& s : scores) { s /= sum; }

        int   top_class = std::max_element(scores.begin(), scores.end()) - scores.begin();
        float top_score = scores[top_class];

        return {scores, top_class, top_score, latency_ms};
    }

private:
    Ort::Env     env_;
    Ort::Session session_{nullptr};
    std::string  input_name_;
    std::string  output_name_;
};

// ─────────────────────────────────────────────────────────────────────────────
// HTTP server endpoints
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Edge-CV Inference Server\n";
    std::cout << "  Model:  " << MODEL_PATH << "\n";
    std::cout << "  Input:  " << INPUT_C << "x" << INPUT_H << "x" << INPUT_W << "\n";
    std::cout << "  Port:   " << SERVER_PORT << "\n\n";

    // ── Load model ────────────────────────────────────────────────────────────
    MappedModel mapped = MappedModel::load(MODEL_PATH);
    InferenceEngine engine(mapped.addr, mapped.size);

    httplib::Server svr;

    // ── GET /health ───────────────────────────────────────────────────────────
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json resp = {{"status", "ok"}, {"model", MODEL_PATH}};
        res.set_content(resp.dump(), "application/json");
    });

    // ── POST /predict ─────────────────────────────────────────────────────────
    // Body: JSON {"pixels": [r,g,b,r,g,b,...], "height": 224, "width": 224}
    //   pixels: flat array of uint8 values in HWC / RGB order
    svr.Post("/predict", [&engine](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);

            if (!body.contains("pixels")) {
                res.status = 400;
                res.set_content(
                    json{{"error", "missing 'pixels' key in request body"}}.dump(),
                    "application/json"
                );
                return;
            }

            int H = body.value("height", INPUT_H);
            int W = body.value("width",  INPUT_W);
            auto pixels_json = body["pixels"].get<std::vector<uint8_t>>();

            if (static_cast<int>(pixels_json.size()) != H * W * 3) {
                res.status = 400;
                res.set_content(
                    json{{"error", "pixels length must be H*W*3"}}.dump(),
                    "application/json"
                );
                return;
            }

            // Preprocess: HWC uint8 → CHW float32 normalized
            auto tensor = preprocess(pixels_json, H, W);

            // Inference
            auto result = engine.run(tensor, H, W);

            // Build response — top-5 classes
            std::vector<int> top5_idx(result.scores.size());
            std::iota(top5_idx.begin(), top5_idx.end(), 0);
            std::partial_sort(
                top5_idx.begin(), top5_idx.begin() + 5, top5_idx.end(),
                [&](int a, int b) { return result.scores[a] > result.scores[b]; }
            );

            json top5 = json::array();
            for (int i = 0; i < std::min(5, (int)top5_idx.size()); ++i) {
                int idx = top5_idx[i];
                top5.push_back({
                    {"class_id", idx},
                    {"score", result.scores[idx]},
                });
            }

            json resp = {
                {"top_class",   result.top_class},
                {"top_score",   result.top_score},
                {"latency_ms",  result.latency_ms},
                {"top5",        top5},
            };
            res.set_content(resp.dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /benchmark ───────────────────────────────────────────────────────
    // Runs N dummy inferences and returns timing statistics.
    // Used by the CI benchmark suite.
    svr.Post("/benchmark", [&engine](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);
            int N = body.value("n", 100);

            // Dummy tensor (zeros — we only care about timing, not accuracy here)
            std::vector<float> dummy(INPUT_C * INPUT_H * INPUT_W, 0.0f);
            std::vector<float> latencies;
            latencies.reserve(N);

            for (int i = 0; i < N; ++i) {
                auto r = engine.run(dummy, INPUT_H, INPUT_W);
                latencies.push_back(r.latency_ms);
            }

            float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
            float mean = sum / N;

            std::sort(latencies.begin(), latencies.end());
            float p50 = latencies[N / 2];
            float p95 = latencies[static_cast<int>(N * 0.95)];
            float p99 = latencies[static_cast<int>(N * 0.99)];

            // Memory usage from /proc/self/status
            long rss_kb = 0;
            std::ifstream proc("/proc/self/status");
            std::string line;
            while (std::getline(proc, line)) {
                if (line.rfind("VmRSS:", 0) == 0) {
                    std::istringstream ss(line.substr(6));
                    ss >> rss_kb;
                    break;
                }
            }

            json resp = {
                {"n",            N},
                {"mean_ms",      mean},
                {"p50_ms",       p50},
                {"p95_ms",       p95},
                {"p99_ms",       p99},
                {"min_ms",       latencies.front()},
                {"max_ms",       latencies.back()},
                {"fps",          1000.0f / mean},
                {"memory_rss_mb", rss_kb / 1024.0},
            };
            res.set_content(resp.dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    std::cout << "Server listening on 0.0.0.0:" << SERVER_PORT << "\n";
    svr.listen("0.0.0.0", SERVER_PORT);
    return 0;
}