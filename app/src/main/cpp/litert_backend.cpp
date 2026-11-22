#include "litert_backend.h"
#include "float_tile_processor.h"
#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>
#include <android/log.h>
#include <sys/system_properties.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>

namespace litert_backend {
namespace {
// Reads a system property as an int, falling back to `fallback` when absent or
// unparsable. Used for on-device A/B tuning of the vendor options below.
int property_int(const char *name, int fallback) {
  char value[PROP_VALUE_MAX] = {};
  int length = __system_property_get(name, value);
  if (length <= 0) return fallback;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  return (end && *end == '\0') ? static_cast<int>(parsed) : fallback;
}

// Raw property text ("" when unset).
std::string property_str(const char *name) {
  char value[PROP_VALUE_MAX] = {};
  int length = __system_property_get(name, value);
  return length > 0 ? std::string(value, static_cast<size_t>(length)) : std::string();
}

// The vendored litert-c-2.2.0 headers predate the opaque-options factory, but
// the bundled libLiteRt.so 2.3.0 exports it. Prototypes match
// litert/c/litert_opaque_options.h at the v2.3.0 tag.
extern "C" {
LiteRtStatus LiteRtCreateOpaqueOptions(const char *payload_identifier, void *payload_data,
                                       void (*payload_destructor)(void *payload_data),
                                       LiteRtOpaqueOptions *options);
void LiteRtDestroyOpaqueOptions(LiteRtOpaqueOptions options);
}

#define API_FUNCTIONS(X) \
  X(LiteRtCreateEnvironment) X(LiteRtDestroyEnvironment) \
  X(LiteRtCreateOptions) X(LiteRtDestroyOptions) X(LiteRtSetOptionsHardwareAccelerators) \
  X(LiteRtCreateOpaqueOptions) X(LiteRtAddOpaqueOptions) X(LiteRtDestroyOpaqueOptions) \
  X(LiteRtCreateModelFromFile) X(LiteRtDestroyModel) X(LiteRtGetModelSignature) \
  X(LiteRtGetNumSignatureInputs) X(LiteRtGetNumSignatureOutputs) \
  X(LiteRtGetSignatureInputTensorByIndex) X(LiteRtGetSignatureOutputTensorByIndex) \
  X(LiteRtGetRankedTensorType) X(LiteRtCreateCompiledModel) X(LiteRtDestroyCompiledModel) \
  X(LiteRtCompiledModelIsFullyAccelerated) X(LiteRtGetCompiledModelInputBufferRequirements) \
  X(LiteRtGetCompiledModelOutputBufferRequirements) X(LiteRtCreateManagedTensorBufferFromRequirements) \
  X(LiteRtDestroyTensorBuffer) X(LiteRtLockTensorBuffer) X(LiteRtUnlockTensorBuffer) \
  X(LiteRtRunCompiledModel)

// The models do not all declare the NHWC 2x/3x shape the caller asks for, so
// the real layout, the graph's own padding border and any integer reduction
// back to the requested page scale are resolved from the declared output
// tensor by mediatek_layout::dispatch (see litert_backend.h).
struct Runtime {
  void *library = nullptr;
#define DECLARE(name) decltype(&name) name##_ = nullptr;
  API_FUNCTIONS(DECLARE)
#undef DECLARE
  LiteRtEnvironment environment = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtModel model = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  LiteRtTensorBuffer input = nullptr, output = nullptr;
  LiteRtRankedTensorType input_type{}, output_type{};
  int scale = 2, padding = 18;
  bool npu = false;
  // Per-model input/output rescale, carried per model instead of a branch in the
  // shared pre/post processor.
  //
  // span-nomosuni-x2 begins with MUL(255) then SUB([0.4488,0.4371,0.404]) and ends
  // with RELU + MINIMUM(1), so the graph was exported for a 0..255 input domain and
  // returns a 0..1 value that is only 1/255 of the page's 0..255 scale. Measured by
  // enumerating (input gain, output gain) against a bicubic reference on page-001
  // (tools: tmp/span_grid.py):
  //     in×1     out×1   PSNR 11.24 dB, output p50 pins to 1.000 (94 % saturated)
  //     in×1/255 out×255 PSNR 25.68 dB, p5/p50/p95 = 0.096/0.943/0.998
  //     (source p5/p50/p95 = 0.102/0.953/0.980)
  // in×1/255 alone gives p50 0.004 — far too dark, which is why both directions are
  // needed. The 11.24 dB case is the grey/ghosted page in
  // logs/116v2/mihon_crash_logs(3).txt:2390/7531/8053 and the SPAN screenshot.
  // Every other model keeps 1.0/1.0, so Real-ESRGAN and W2xEX are untouched.
  float input_scale = 1.0f, output_scale_factor_ = 1.0f;
  mediatek_layout::Dispatch plan;
  std::string key, error, library_directory, cache_directory;

  bool fail(const std::string &message) {
    error = message;
    __android_log_print(ANDROID_LOG_ERROR, "LiteRtBackend", "%s", message.c_str());
    return false;
  }
  bool okay(LiteRtStatus status, const char *stage) {
    return status == kLiteRtStatusOk || fail(std::string(stage) + ": " + std::to_string(status));
  }
  void close() {
    if (input) LiteRtDestroyTensorBuffer_(input);
    if (output) LiteRtDestroyTensorBuffer_(output);
    if (compiled) LiteRtDestroyCompiledModel_(compiled);
    if (model) LiteRtDestroyModel_(model);
    if (options) LiteRtDestroyOptions_(options);
    if (environment) LiteRtDestroyEnvironment_(environment);
    input = output = nullptr;
    compiled = nullptr; model = nullptr; options = nullptr; environment = nullptr;
    npu = false; key.clear(); plan = mediatek_layout::Dispatch{};
    if (library) dlclose(library);
    library = nullptr;
  }
  // All 15 models take NHWC FP32 [1,256,256,3]. Reject anything else loudly.
  static bool check_input(const LiteRtRankedTensorType &type) {
    const auto &l = type.layout;
    return type.element_type == kLiteRtElementTypeFloat32 && l.rank == 4 &&
        !l.has_strides && l.dimensions[0] == 1 &&
        l.dimensions[1] == mediatek_layout::kInputSide &&
        l.dimensions[2] == mediatek_layout::kInputSide && l.dimensions[3] == 3;
  }
  bool load(const std::string &path, const std::string &cache, const std::string &libraries,
            int output_scale, int border, bool cpu_test) {
    std::string identity = path + "|" + cache + "|" + libraries + "|" + std::to_string(output_scale) + "|" + std::to_string(border) + (cpu_test ? "|cpu-test" : "|npu");
    if (compiled && key == identity) return true;
    close(); error.clear();
    if (output_scale != 2 && output_scale != 3) return fail("Unsupported model scale");
    // span-nomosuni-x2 (the 82-op rewrite, first op SUB) normalises exactly like the
    // pretrained SPAN: it takes the app's [0,1] tile and subtracts the ImageNet-style
    // mean [0.4488,0.4371,0.404]. Its output, however, is a 0..1 fraction of a
    // 0..255 page, i.e. 1/255 of the shared contract, so it needs output_scale 255.
    //
    // Measured on page-001 against a bicubic reference (tmp/r3_span_bug.py):
    //   in×1     out×1    p50 0.0037, PSNR  1.79 dB   <- near-black page
    //   in×1/255 out×255  p50 0.0082, PSNR  1.79 dB   <- also near-black
    //   in×1     out×255  p50 0.9431, PSNR 25.68 dB   <- correct
    //   source            p50 0.9531
    // The input must stay [0,1]; only the output is rescaled. An earlier revision of
    // this file also divided the input by 255, which produced the black SPAN page.
    const bool span_domain = path.find("span-nomosuni") != std::string::npos;
    input_scale = 1.0f;
    output_scale_factor_ = span_domain ? 255.0f : 1.0f;
    library = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) return fail("Unable to load LiteRT runtime");
#define LOAD(name) name##_ = reinterpret_cast<decltype(name##_)>(dlsym(library, #name)); \
    if (!name##_) return fail("Missing LiteRT API: " #name);
    API_FUNCTIONS(LOAD)
#undef LOAD
    LiteRtEnvOption env_options[3]{};
    const LiteRtEnvOptionTag tags[] = {kLiteRtEnvOptionTagCompilerPluginLibraryDir,
        kLiteRtEnvOptionTagDispatchLibraryDir, kLiteRtEnvOptionTagCompilerCacheDir};
    library_directory = libraries; cache_directory = cache;
    const char *values[] = {library_directory.c_str(), library_directory.c_str(), cache_directory.c_str()};
    for (int i = 0; i < 3; ++i) {
      env_options[i].tag = tags[i];
      env_options[i].value.type = kLiteRtAnyTypeString;
      env_options[i].value.str_value = values[i];
    }
    if (!okay(LiteRtCreateEnvironment_(3, env_options, &environment), "Create environment") ||
        !okay(LiteRtCreateOptions_(&options), "Create options") ||
        !okay(LiteRtSetOptionsHardwareAccelerators_(options, cpu_test ?
            kLiteRtHwAcceleratorCpu : kLiteRtHwAcceleratorNpu), "Select accelerator") ||
        !okay(LiteRtCreateModelFromFile_(environment, path.c_str(), &model), "Load model")) return false;
    // Attach MediaTek vendor compile options. The vendor plugin looks up a
    // TOML-style payload under the identifier "mediatek"
    // (litert/c/options/litert_mediatek_options.cc: LrtGetOpaqueMediatekOptionsData
    // encodes `key = value` lines and returns that identifier) and maps
    //   performance_mode  -> NeuronCompilation_setPreference
    //   optimization_hint -> NeuronCompilation_setOptimizationHint
    // (vendor symbols verified in libneuronusdk_adapter.mtk.so 8.2.26). Without
    // this payload the plugin compiles with no preference and hint at all.
    //
    // Values (litert_mediatek_options.h):
    //   performance_mode: 0 LowPower, 1 FastSingleAnswer, 2 SustainedSpeed, 3 TurboBoost
    //   optimization_hint bitfield: 1 LowLatency, 2 DeepFusion, 4 BatchProcessing
    // Defaults request the max-clock TurboBoost preference plus
    // LowLatency|DeepFusion; neither touches model math, only scheduling and
    // QoS, so the NPU-only contract and the numerics are unchanged. Both are
    // overridable for on-device A/B via system properties:
    //   debug.mihon.npu.perfmode  (0 disables the preference line)
    //   debug.mihon.npu.hint      (0 disables the hint line)
    if (!cpu_test) {
      const int perf_mode = property_int("debug.mihon.npu.perfmode", 3);
      const int opt_hint = property_int("debug.mihon.npu.hint", 3);
      std::string payload;
      if (perf_mode > 0) payload += "performance_mode = " + std::to_string(perf_mode) + "\n";
      if (opt_hint > 0) payload += "optimization_hint = " + std::to_string(opt_hint) + "\n";
      // Escape hatch for on-device experiments with the remaining documented
      // keys (e.g. 'l1_cache_optimizations = true', 'option_bundle = "..."');
      // unknown keys are ignored by the vendor parser.
      payload += property_str("debug.mihon.npu.extra");
      if (!payload.empty()) {
        char *data = static_cast<char *>(std::malloc(payload.size() + 1));
        if (data) {
          std::memcpy(data, payload.c_str(), payload.size() + 1);
          LiteRtOpaqueOptions opaque = nullptr;
          auto release_data = [](void *p) { std::free(p); };
          if (okay(LiteRtCreateOpaqueOptions_("mediatek", data, release_data, &opaque),
                   "Create mediatek options")) {
            if (okay(LiteRtAddOpaqueOptions_(options, opaque), "Add mediatek options")) {
              __android_log_print(ANDROID_LOG_INFO, "LiteRtBackend",
                                  "MediaTek NPU compile options injected: %s",
                                  payload.c_str());
            } else {
              LiteRtDestroyOpaqueOptions_(opaque);
            }
          } else {
            std::free(data);
          }
        }
      }
    }
    LiteRtSignature signature = nullptr;
    LiteRtTensor in_tensor = nullptr, out_tensor = nullptr;
    LiteRtParamIndex input_count = 0, output_count = 0;
    if (!okay(LiteRtGetModelSignature_(model, 0, &signature), "Read signature") ||
        !okay(LiteRtGetNumSignatureInputs_(signature, &input_count), "Count inputs") ||
        !okay(LiteRtGetNumSignatureOutputs_(signature, &output_count), "Count outputs")) return false;
    if (input_count != 1 || output_count != 1) return fail("Expected one input and output");
    if (!okay(LiteRtGetSignatureInputTensorByIndex_(signature, 0, &in_tensor), "Input tensor") ||
        !okay(LiteRtGetSignatureOutputTensorByIndex_(signature, 0, &out_tensor), "Output tensor") ||
        !okay(LiteRtGetRankedTensorType_(in_tensor, &input_type), "Input type") ||
        !okay(LiteRtGetRankedTensorType_(out_tensor, &output_type), "Output type")) return false;
    if (!check_input(input_type))
      return fail("Expected a contiguous FP32 NHWC [1,256,256,3] input tensor");
    const auto &out_layout = output_type.layout;
    if (output_type.element_type != kLiteRtElementTypeFloat32 || out_layout.rank != 4 ||
        out_layout.has_strides)
      return fail("Expected a contiguous FP32 rank-4 output tensor");
    plan = mediatek_layout::dispatch(out_layout.dimensions, out_layout.rank, output_scale);
    if (!plan.ok) return fail(plan.error);
    // float_tile_processor.h sizes every tile as output_tile = 256*scale and rejects any other
    // length, so a geometry that does not land exactly on that side must never be accepted.
    if (plan.target_side != 256 * output_scale)
      return fail("Model output side does not match the float_tiles tile geometry");
    if (!okay(LiteRtCreateCompiledModel_(environment, model, options, &compiled), "Compile model")) return false;
    bool fully_accelerated = false;
    if (!okay(LiteRtCompiledModelIsFullyAccelerated_(compiled, &fully_accelerated), "Check delegation")) return false;
    // Request NPU only and reject any CPU/partial delegation. Never label a CPU model as NPU.
    if (!cpu_test && !fully_accelerated) {
      LiteRtDestroyCompiledModel_(compiled); compiled = nullptr;
      return fail("The model could not be fully delegated to MediaTek NPU; enhancement skipped");
    }
    LiteRtTensorBufferRequirements in_requirements = nullptr, out_requirements = nullptr;
    if (!okay(LiteRtGetCompiledModelInputBufferRequirements_(compiled, 0, 0, &in_requirements), "Input requirements") ||
        !okay(LiteRtGetCompiledModelOutputBufferRequirements_(compiled, 0, 0, &out_requirements), "Output requirements") ||
        !okay(LiteRtCreateManagedTensorBufferFromRequirements_(environment, &input_type, in_requirements, &input), "Input buffer") ||
        !okay(LiteRtCreateManagedTensorBufferFromRequirements_(environment, &output_type, out_requirements, &output), "Output buffer")) return false;
    key = identity; scale = output_scale; padding = border; npu = !cpu_test;
    __android_log_print(ANDROID_LOG_INFO, "LiteRtBackend",
                        "Ready: %s (%s) layout=%s declared=%d crop=%d downsample=%d target=%d",
                        path.c_str(), npu ? "fully delegated NPU" : "CPU validation only",
                        plan.layout == mediatek_layout::Layout::Nchw ? "NCHW" : "NHWC",
                        plan.output_side, plan.crop, plan.downsample, plan.target_side);
    return true;
  }
  // Produces exactly float_tiles' NHWC (256*scale)^2*3 buffer from whichever
  // layout the graph declared, ignoring the graph's padding border and
  // reducing a 4x-native result to the requested 2x page scale.
  // Converts the model's declared output into float_tiles' NHWC (256*scale)^2*3
  // buffer, honouring the graph's padding border and any integer downsample.
  //
  // `write_side` is how many rows/columns float_tiles will actually copy
  // (core*scale = (256-2*padding)*scale, i.e. 440 for Real-CUGAN at padding 18 or
  // 432 at padding 20). The remainder of the 512-wide buffer is never read: the
  // hard copy in float_tile_processor.h indexes at most core*scale-1 on each axis.
  // Computing only that square removes (512^2-440^2)/512^2 = 26.1 % of the work
  // per tile at padding 18, and this is a pure host-side loop over floats that
  // runs once per tile (35 times a page), so it is a straight win with no change
  // to any pixel value.
  void convert(const float *declared, float *out, int write_side) const {
    const int side = plan.output_side, crop = plan.crop, factor = plan.downsample;
    const int stride = plan.target_side;   // float_tiles' buffer is (256*scale) wide
    const int limit = std::min(stride, std::max(0, write_side));
    const bool nchw = plan.layout == mediatek_layout::Layout::Nchw;
    const size_t plane = static_cast<size_t>(side) * side;
    const float weight = 1.0f / static_cast<float>(factor * factor);
    if (nchw && factor == 2) {
      // Streaming fast path for the NCHW 4x-native graphs (Real-ESRGAN): walk
      // each channel plane front to back, keeping only two 4 KB source rows in
      // cache at a time, instead of visiting all three planes per output pixel.
      // Same values, same order of accumulation (dy outer, dx inner), so the
      // box average is bit-identical to the generic loop below.
      for (int c = 0; c < 3; ++c) {
        const float *base = declared + static_cast<size_t>(c) * plane +
                            static_cast<size_t>(crop) * side + crop;
        for (int y = 0; y < limit; ++y) {
          const float *row0 = base + static_cast<size_t>(y * 2) * side;
          const float *row1 = row0 + side;
          float *dst = out + (static_cast<size_t>(y) * stride) * 3 + c;
          for (int x = 0; x < limit; ++x) {
            const float *p0 = row0 + x * 2, *p1 = row1 + x * 2;
            dst[static_cast<size_t>(x) * 3] = (p0[0] + p0[1] + p1[0] + p1[1]) * weight;
          }
        }
      }
      return;
    }
    if (nchw && factor == 1) {
      // Same idea for the native-2x NCHW graphs (Real-CUGAN / W2xEX): one
      // sequential sweep per channel plane, interleaved stores to the NHWC
      // output. Value order is a straight copy, so the result is identical.
      for (int c = 0; c < 3; ++c) {
        const float *base = declared + static_cast<size_t>(c) * plane +
                            static_cast<size_t>(crop) * side + crop;
        for (int y = 0; y < limit; ++y) {
          const float *src = base + static_cast<size_t>(y) * side;
          float *dst = out + (static_cast<size_t>(y) * stride) * 3 + c;
          for (int x = 0; x < limit; ++x) {
            dst[static_cast<size_t>(x) * 3] = src[x];
          }
        }
      }
      return;
    }
    for (int y = 0; y < limit; ++y) {
      for (int x = 0; x < limit; ++x) {
        const int sx = crop + x * factor, sy = crop + y * factor;
        for (int c = 0; c < 3; ++c) {
          float value = 0;
          for (int dy = 0; dy < factor; ++dy) {
            const size_t row = static_cast<size_t>(sy + dy) * side;
            for (int dx = 0; dx < factor; ++dx) {
              const size_t pixel = row + sx + dx;
              value += declared[nchw ? c * plane + pixel : pixel * 3 + c];
            }
          }
          out[(static_cast<size_t>(y) * stride + x) * 3 + c] = value * weight;
        }
      }
    }
  }
  bool run(const std::vector<float> &data, std::vector<float> &result) {
    if (!compiled || !input || !output || !plan.ok || data.size() != 256*256*3) return fail("Model is not ready");
    // Optional per-tile stage timing (debug.mihon.npu.tiletiming=1): separates
    // the host-side buffer copies/conversion from the NPU execution itself.
    const bool tile_timing = property_int("debug.mihon.npu.tiletiming", 0) == 1;
    const auto t_begin = std::chrono::steady_clock::now();
    void *memory = nullptr;
    if (!okay(LiteRtLockTensorBuffer_(input, &memory, kLiteRtTensorBufferLockModeWrite), "Lock input")) return false;
    if (input_scale == 1.0f) {
      std::memcpy(memory, data.data(), data.size() * sizeof(float));
    } else {
      // Copy-scaled so the caller's tile vector stays in the app-wide [0,1] form.
      float *dst = static_cast<float *>(memory);
      for (size_t i = 0; i < data.size(); ++i) dst[i] = data[i] * input_scale;
    }
    if (!okay(LiteRtUnlockTensorBuffer_(input), "Unlock input")) return false;
    const auto t_input = std::chrono::steady_clock::now();
    if (!okay(LiteRtRunCompiledModel_(compiled, 0, 1, &input, 1, &output), "Execute model")) return false;
    const auto t_execute = std::chrono::steady_clock::now();
    if (!okay(LiteRtLockTensorBuffer_(output, &memory, kLiteRtTensorBufferLockModeRead), "Lock output")) return false;
    // Exactly (256*scale)^2*3 NHWC floats: float_tiles asserts this length and indexes by it.
    const size_t target = static_cast<size_t>(plan.target_side) * plan.target_side * 3;
    if (plan.target_side != 256 * scale) {
      LiteRtUnlockTensorBuffer_(output);
      return fail("Model output geometry does not match the active tile scale");
    }
    result.resize(target);
    // float_tiles reads this buffer at index padding*scale + core_x for
    // core_x in [0, core*scale), so the highest row/column it ever touches is
    // padding*scale + core*scale - 1 = scale*(256 - padding) - 1. Only that
    // square needs computing; at padding 18 / scale 2 that is 476 of 512 rows,
    // i.e. (512^2 - 476^2)/512^2 = 13.6 % less host work per tile, and at
    // padding 20 it is 472. Pixels beyond it stay at the vector's zero fill and
    // are never read.
    const int write_side = scale * (256 - padding);
    convert(static_cast<const float *>(memory), result.data(), write_side);
    // SPAN's graph returns a 0..1 fraction of a 0..255 page, i.e. its numbers are
    // 1/255 of the shared [0,1] contract. Scaling here (and in the input copy
    // above) keeps every other model on the original numbers. float_tiles clamps
    // to [0,1] afterwards, so the occasional 1.10 overshoot saturates to white
    // exactly as the graph's own MINIMUM(1) would.
    if (output_scale_factor_ != 1.0f) {
      for (size_t i = 0; i < result.size(); ++i) result[i] *= output_scale_factor_;
    }
    if (!okay(LiteRtUnlockTensorBuffer_(output), "Unlock output")) return false;
    if (tile_timing) {
      using ms = std::chrono::duration<double, std::milli>;
      const auto t_end = std::chrono::steady_clock::now();
      __android_log_print(ANDROID_LOG_INFO, "LiteRtBackend",
                          "tile: total=%.1f input=%.1f execute=%.1f post=%.1f",
                          ms(t_end - t_begin).count(), ms(t_input - t_begin).count(),
                          ms(t_execute - t_input).count(), ms(t_end - t_execute).count());
    }
    return true;
  }
};
Runtime runtime;
} // namespace

bool initialize(const std::string &model, const std::string &cache, const std::string &libraries,
                int scale, int padding, bool cpu_test_only) {
  if (runtime.load(model, cache, libraries, scale, padding, cpu_test_only)) return true;
  runtime.close();
  return false;
}
bool is_initialized() { return runtime.compiled && runtime.input && runtime.output; }
bool is_npu_active() { return is_initialized() && runtime.npu; }
std::string last_error() { return runtime.error; }
bool run_tile(const std::vector<float> &input, std::vector<float> &output) { return runtime.run(input, output); }
int process_rgba(const uint8_t *input, int width, int height, int input_stride,
                 uint8_t *output, int output_stride, periodic_texture_guard::Analysis &analysis,
                 std::atomic<int> *progress, const std::atomic<bool> *abort) {
  if (!is_initialized()) return -1;
  // runtime.scale stays the caller's requested factor so float_tiles keeps the
  // established 2x/3x page geometry; run() already reduced a 4x graph result
  // back to that factor.
  return float_tiles::process(input, width, height, input_stride, output, output_stride,
      runtime.scale, runtime.padding, analysis, progress, abort,
      [](const std::vector<float> &in, std::vector<float> &out) { return runtime.run(in, out); });
}
void shutdown() { runtime.close(); }
} // namespace litert_backend
