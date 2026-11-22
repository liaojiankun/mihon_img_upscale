# MediaTek NPU backend

The reader offers MediaTek NPU on Android 12+ arm64 MediaTek devices. It uses the official
LiteRT 2.3.0 runtime and the NeuroPilot compiler/dispatch Maven artifacts
(`litert-npu-runtime-mediatek:2.3.0`). The 15 NPU model variants ship as fixed-shape TFLite
models with **FLOAT32 weights** and FP32 NHWC RGB input/output. Qualcomm keeps its separate QNN DLC
and serialized-context path.

Float32 weights are not a style choice. An FP16 export (`Optimize.DEFAULT` +
`supported_types=[tf.float16]`) leaves every conv filter and bias as the output of a `DEQUANTIZE`
op, i.e. a non-constant tensor, and MediaTek's compiler refuses exactly that — see
[What the on-device log proved](#what-the-on-device-log-proved).

An on-device `logcat` from a Dimensity 9400+ now shows all five distinct graphs compiled and
executed as `fully delegated NPU`, one partition each — see
[What the on-device log proved](#what-the-on-device-log-proved). The remaining defects were not
compilation failures: they were a tile-seam artifact, a W2xEX page that never reached the NPU, and a
SPAN model whose input contract did not match the caller's. All three are described in
[Inference correctness](#inference-correctness).

Selecting a model prepares it on an IO worker. The model SHA-256, runtime/cache version, Android
build fingerprint and ABI identify an app-private directory in `noBackupFilesDir/mediatek-npu`.
The verified TFLite model and the LiteRT compiler cache remain there across launches. LiteRT also
invalidates its compiled artifacts when its compiler or compilation options change. Image-cache
eviction does not remove prepared models. Concurrent reader/UI preparation is serialized, and a
failed preparation requires an explicit retry during that process.

Preparation uses a temporary native runtime. Reader initialization loads the chosen model from the
same persistent compiler cache. Native inference shares the reader's existing output, monochrome
protection and alpha handling, with fixed 256-pixel input tiles and each model's established
overlap. Device libraries are declared optional in the manifest, so their absence does not prevent
installation.

Both NPU backends follow the same failure policy: **skip enhancement and retain the original
image**. The stored backend choice is preserved. MediaTek requests NPU only and rejects a compiled
graph unless `LiteRtCompiledModelIsFullyAccelerated` confirms complete acceleration. There is no
automatic CPU or Vulkan fallback. Vulkan runs only when explicitly selected. Failed native
inference returns null, so the original cannot be saved as a successful enhanced result. Input
prescaling uses a temporary bitmap and preserves the original for display on failure.

An NPU page does not depend on the ncnn model assets at all: backend resolution happens before any
asset extraction, and a non-Vulkan backend takes the geometry-only JNI entry instead. A missing or
undownloadable `.param`/`.bin` therefore cannot turn an NPU page back into the original image, which
is what used to happen to W2xEX offline.

## Why the models are what they are

MediaTek's compiler has a hard operator allow-list. The authoritative table is
`MediaTek_Neuro_Compiler.md` from [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)
(archived locally at `tools/downloads/mtk_compiler.md`). It contains exactly 47 TFLite ops plus the
`odml.rms_norm` / `odml.l2_norm` / `odml.group_norm` composites. Three ops that the source graphs
used are **not** on it:

| Op | Reaches the graph from | Rewritten to |
| :--- | :--- | :--- |
| `MIRROR_PAD` | `onnx2tf` conversion of the Real-CUGAN Pro padding mode | `PAD` |
| `GATHER` | `onnx2tf` channel permutation ahead of the pixel shuffle | `TRANSPOSE` |
| `DEPTH_TO_SPACE` | the pixel-shuffle tail of Real-ESRGAN / SPAN / W2xEX | `RESHAPE` + `TRANSPOSE` |

Being on the allow-list is necessary but not sufficient. The on-device log then showed two further
rejections that the allow-list says nothing about — tensor rank and weight constness. Both are now
structural properties of the rewrite, not per-model fixes:

- **Every tensor rank ≤ 4.** `RESHAPE` tolerates rank 5, but `TRANSPOSE` does not: three of
  Neuron's four backends demand 4D. The pixel-shuffle therefore decomposes into a chain whose
  *transposes* are all rank 4 and whose reshapes are all rank ≤ 4. The earlier rank-5 build only
  silenced the `RESHAPE` complaint and left the `TRANSPOSE` one, which is why it was not enough.
  `tools/mtk_ops/d2s_rank4.py` proves the chain bit-exact (maxdiff `0.000e+00`) against the numpy
  `DepthToSpace` definition over 64 combinations: CRD/DCR × N=2,4 × four channel counts × two
  spatial sizes × NCHW and NHWC.
- **Every conv filter and bias is a constant buffer.** Export uses float32 weights with no
  `Optimize.DEFAULT`; `DEQUANTIZE` count is 0 in all 15 models.

The rewritten ONNX graphs live in `conversion/onnx_mtk/`; the export is
`tools/mtk_ops/convert_mtk.py` (`onnx2tf` flatbuffer-direct → SavedModel → float32 TFLite). Each of
the 15 resulting models now uses only allow-listed ops — measured, not assumed:

| Model | Variants | Output | Ops | Size (B) |
| :--- | :--- | :--- | ---: | ---: |
| Real-CUGAN SE 2x | no-denoise, denoise 1/2/3, conservative | `[1,3,512,512]` | 100–108 | 5 161 432–5 161 832 |
| Real-CUGAN Pro 2x | no-denoise, denoise 3, conservative | `[1,3,512,512]` | 98 | 5 459 468 |
| Real-CUGAN Pro 3x | no-denoise, denoise 3, conservative | `[1,3,792,792]` | 98 | 5 466 380 |
| Real-ESRGAN animevideov3 | standard | `[1,3,1024,1024]` | 97 | 2 504 108 |
| Real-ESRGAN general-x4v3 | standard | `[1,3,1024,1024]` | 177 | 4 885 872 |
| SPAN NomosUni | standard | `[1,512,512,3]` | 84 | 1 659 928 |
| W2xEX photo-small | standard | `[1,3,512,512]` | 95 | 2 420 700 |

Input is `[1,256,256,3]` float32 in every case. Total 70 056 512 B (was 35 304 376 B with FP16
weights — the models roughly doubled, which is the price of constant float32 weights). SPAN carries
one extra `MUL` for the input-scale adapter described in [Inference correctness](#inference-correctness).

The output *declared* layout differs per family — NCHW for most, NHWC for SPAN — and the 3x / 4x
models declare a padded side (792 = 3·(256+2·4)). `mediatek_layout::dispatch()` in
`app/src/main/cpp/litert_backend.h` derives layout, native scale, crop and integer downsample from
the declared dims, and both `load()` and `run()` hard-gate `plan.target_side == 256 * scale`
because `float_tile_processor.h` sizes every tile as exactly that.

## Inference correctness

Compiling is not the same as producing the right image. Three defects survived the first successful
NPU run and were located with the real TFLite models offline.

### Tile seams (banding on Real-CUGAN SE / Pro output)

The on-device screenshot showed a **1–3 pixel dark line every 376 screen rows** across both artwork
and the flat white margin, i.e. exactly on the tile grid: `core * scale = (256 - 2*18) * 2 = 440`
bitmap rows, displayed at 0.8548× as 376 screen rows. Adjacent tiles were measured to disagree by up
to **0.0985 (25 codes)** on the rows they both predict, at padding 18, decaying monotonically with
padding (`tools/mtk_ops/overlap_scan.py`: 0.00330 at pad 8 → 0.00066 at pad 96). The model itself
has no border defect — on a constant field the deviation is 0.0000–0.0007 over zero rows
(`tools/mtk_ops/edge_effect.py`) — so the step is context truncation at the hard cut, not a model bug.

`float_tile_processor.h` now feathers each tile into the neighbour it overlaps: `value_at(x, y, c)`
returns the tile's own prediction, and before the hard copy the padding band it shares with the
already-written top/left neighbour is ramped in with `alpha = i / (feather - 1)` over
`feather = min(padding*scale, cw, ch)`.

A texture-guarded neighbour (`periodic_texture_guard`, enabled only for Real-CUGAN Pro on neutral
pages) is resampled through the 3/4 shrink-restore transform, so its geometry does not line up with
this tile's band; those seams are deliberately left on the old path via `plan.guards(...)`.

Measured by faithfully replaying the C++ order in Python against the real model
(`tools/mtk_ops/feather_sim.py`, `line_metric.py`):

| Metric at seam rows | Hard cut | Feathered |
| :--- | ---: | ---: |
| Step (6 rows either side) | 0.68 | **0.34** codes |
| Line depth | 0.28 | **0.09** codes (−68 %) |
| Worst single-row jump | 14.54 | **11.04** codes |
| Change outside the feather bands | — | 0.024 codes |

Not reproduced: the exact 1–3 row dip in the reported screenshot. Replaying that page through the
real model whitened the margin to 255 before any seam could form, so the artifact's absolute depth is
not reproducible offline; only the quantity that generates it (neighbour disagreement) is. Treat the
on-device result as the acceptance test.

### W2xEX pages never reached the NPU

`initW2xEx` called `extractModelsToCache(context, selectedModel.assetPath) ?: return false` *before*
resolving the backend. The W2xEX ncnn weights are not bundled, so that call goes to Hugging Face; on a
device without connectivity it threw `SocketTimeoutException … after 15000ms` from
`Waifu2x.downloadDirectModels(Waifu2x.kt:933)` via `initW2xEx(Waifu2x.kt:513)`, the whole page
returned false, and the already-prepared NPU model was never used.

Backend resolution now happens first. When the resolved backend is not Vulkan, `initW2xEx` delegates
to `initBackendOnly(...)`, which calls the new JNI entry `nativeInitBackendOnly(scale, padding)` — a
geometry-only `Waifu2x` instance (gpuid `-1`, no Vulkan device, no `.param`/`.bin` load) — and then
`initializeBackend(...)`. The ncnn asset path is not touched at all. `isW2xExInitialized` is only set
once the backend accepts the model, so a failure still leaves every `process*` entry point returning
null and the reader showing the original.

Covered by `MediaTekNpuDeviceTest.w2xExNpuPageSkipsAssetExtraction`, which asserts the call fails in
under 5 s (the old path needed a 15 s-per-file download timeout) and that no other backend runs.

### SPAN output is black — input contract mismatch

The app feeds every model `[0, 1]` (`float_tile_processor.h:39`, `input[...] / 255.0f`). The SPAN
graph expects `[0, 255]`. Feeding four candidate conventions through both the rewritten ONNX and the
installed TFLite (`tools/mtk_ops/span_probe.py`) gives:

| Input convention | ONNX output | TFLite output |
| :--- | :--- | :--- |
| as-is `[0,1]` | max 0.00549, mean 0.00223 | max 0.00549, mean 0.00223 |
| `x255` | max 1.00000, mean 0.76984 | max 1.00000, mean 0.76986 |

`0.00549` is 1.4/255 — the black page. Not a compile failure (that model logged
`fully delegated NPU` too) and not a layout misread: `span_localize.py` shows the old/new disagreement
spread evenly over channels, rows and columns with no 2×2 lattice, channel permutation or shift.

The fix is a float32 `Mul(255)` prepended inside the SPAN graph, so the graph adapts to the app-wide
contract instead of the shared C++ preprocessor special-casing one of fifteen models. Source graph:
`conversion/onnx_mtk/span-nomosuni-2x_nhwc_255.onnx`; export log
`tools/mtk_ops/export_span255.log`; installed model `mediatek-models/span-nomosuni-x2.tflite`
(1 659 928 B, 84 ops, `[1,256,256,3]` float32 in, `[1,512,512,3]` float32 out, `DEQUANTIZE = 0`,
max rank 4, allow-list clean).

`tools/mtk_ops/span_vs_prev.py` proves the adapter is faithful: new model fed `[0,1]` versus the
previous model fed `x255` gives **maxdiff 0.000e+00** for random, flat and gradient inputs, while the
previous model fed `[0,1]` reproduces the reported black output.

## Performance (Real-ESRGAN on Dimensity 9400+)

Assessed, not changed. The device takes the **JIT** compile path (`soc_model=(null)` in logcat), and
LiteRT 2.3.0's JIT hardcodes `--opt-accuracy --opt 3` with no `--num-mdla`, so APU SMP is off and
`AotCompilationOptions` is ignored. `compile_model.cc:104-113` unconditionally calls
`relax_fp32_to_fp16(model, true)`, meaning **fp16 compute already happens** and no user-side precision
switch exists — re-exporting fp16 weights is a different thing and stays rejected, because it turns
conv filters into `DEQUANTIZE` outputs (see above).

Tiling arithmetic for a 1114×1600 page at padding 18: tile 256 → 48 tiles, 3 145 728 NPU input
pixels, **+76.5 % pure padding overhead**. The criterion is `min(ceil(W/u)·ceil(H/u)·u²)`, not
"bigger is better": tile 384 → 20 tiles (−6.2 %), tile 448 → 12 tiles (best measured).

Ranked candidates, all **not yet measured on device**:

| # | Change | Expected | Cost | Status |
| ---: | :--- | :--- | :--- | :--- |
| 1 | Dynamic tile 384/448 chosen by the arithmetic above | 76.5 % padding → ~0, ~20–35 % end-to-end | re-export 15 models at `[1,384,384,3]`/`[1,448,448,3]`, relax the `tile = 256` and `plan.target_side == 256*scale` gates | arithmetic verified, unmeasured |
| 2 | Ship AOT bytecode in assets | removes first-open Neuron compile | per-SoC/firmware artifacts, invalidated by OTA | unmeasured (JIT ignores AOT options) |
| 3 | `--num-mdla=2 --opt 3` via AOT | APU SMP on | needs the AOT path | unmeasured |
| 4 | `PerformanceMode=TurboBoost` + `OptimizationHint=LowLatency` | 5–15 % | power/heat | unmeasured |
| 5 | Per-thread `CompiledModel` | near-linear page-level throughput | VRAM × concurrency (heap budget 10 803 MB) | unmeasured |
| 6 | Batched `[N,256,256,3]` export | less per-Run overhead | tail-batch waste, dynamic batch | unmeasured |
| 7 | Check `compilation_get_input_padded_size` == 786432 | diagnostic only | — | unmeasured |

Deliberately not pursued: int8 PTQ/QAT (measured 24–32 dB PSNR loss) and flipping
`relax_fp32_to_fp16` (would make Neuron reject the weights again).

## Precision

`README`'s Qualcomm path offers INT8 (W8A8) for one model. MediaTek's supported data-type table is
Float32 / Float16 / Int32 / Int16 / UInt8 / Int8 / Int4 / Int2 / Bool / Int64 — **there is no
Float8**, which is why the earlier FP8 integration was removed rather than kept as a dead gate.
Int8 maps to `NEURON_TENSOR_QUANT8_SYMM` / `QUANT8_ASYMM_SIGNED`, per-tensor or per-channel, so
NeuroPilot does accept it.

It is **not** used, because post-training quantization of these graphs measurably destroys output
quality. Measured against the shipped FP16 models on the same input:

| Model | Full-integer QDQ (int8 activations + weights) | Weight-only int8 |
| :--- | :--- | :--- |
| Real-CUGAN SE 2x no-denoise | maxdiff 2.21e-1, PSNR 29.47 dB | maxdiff 3.38e-1, PSNR 24.80 dB |
| Real-CUGAN Pro 2x conservative | — | maxdiff 1.56e-1, PSNR 32.11 dB |
| Real-ESRGAN general-x4v3 | — | maxdiff 3.56e-1, PSNR 24.10 dB |
| Real-CUGAN Pro 3x no-denoise | — | maxdiff 2.48e-1, PSNR 26.95 dB |
| W2xEX photo-small | — | maxdiff 4.79e-1, PSNR 24.60 dB |
| SPAN NomosUni | — | maxdiff 1.16e-2, PSNR 55.31 dB |

For reference, the FP16 models' own error floor against the FP32 ONNX graphs is 2.1e-3 maxdiff
(120 dB PSNR) — so the quantized variants are ~100× worse. 24–32 dB PSNR is visible degradation in
a super-resolution output, so INT8 is not offered as a selectable or default precision. Matching the
Qualcomm W8A8 path would need quantization-aware retraining, which is out of scope here.

`MediaTekNpu.DEFAULT_PRECISION = PRECISION_FP16`, and `MediaTekNpu.supportedPrecisions()` returns
only `PRECISION_FP16`. `PRECISION_INT8 = 2` exists as a documented constant (the same value
`QnnModelContexts.PRECISION_W8A8` uses) so a future INT8 export can be wired in without a
vocabulary change, but no code path selects it and no INT8 model is packaged.

## Export and build

The models are committed under `mediatek-models/` and staged into `assets/litert-models` together
with a SHA-256 `manifest.json` by the `stageMediaTekModels` Gradle task
(`app/build.gradle.kts:126`). The directory defaults to `mediatek-models` and can be overridden with
`mediatek.tflite.dir` in `local.properties`, `-PmediatekTfliteDir=...`, or `MEDIATEK_TFLITE_DIR`.
A missing file fails the build rather than shipping an incomplete asset set.

Re-exporting (a Python 3.11+ environment with `tensorflow`, `onnx`, `onnxruntime` and `onnx2tf`):

```sh
python tools/mtk_ops/convert_mtk.py            # all 15
python tools/mtk_ops/convert_mtk.py realcugan_pro_2x_conservative
```

The export verifies, per model: the op set against the allow-list, the declared output shape
against the app's contract, `max_rank <= 4`, `DEQUANTIZE == 0` for conv-like ops (filter/bias must
be constant), and numerics against both the previously shipped model and the FP32 ONNX reference.
It only installs a model when all of those pass. Reconcile with:

```sh
python tools/mtk_ops/verify_numerics.py   # each model vs its FP32 ONNX reference
python tools/mtk_ops/cmp_baseline.py      # each model vs mihon-main's turn-1 exports
python tools/mtk_ops/audit_delegation.py  # rank / dynamic-weight / op audit
python tools/mtk_ops/d2s_rank4.py         # rank-4 pixel-shuffle proof vs numpy DepthToSpace
```

`conversion/mediatek-models.orig/` holds a byte-exact snapshot of the pre-rewrite FP16 models,
taken before the first conversion wrote anything; all 15 files compare identical to
`mihon-main/enhancer/src/main/assets/models/*.tflite`. Either directory works as a baseline — it is
what `not_worse` in the export gate measures against.

## Validation status

Verified on this workstation:

- 15/15 models parse, declare `[1,256,256,3]` input, and declare the output shape the app's
  `mediatek_layout::dispatch()` expects.
- 15/15 models use only allow-listed ops, so full delegation is *possible* per the vendor table.
- 15/15 models satisfy the two constraints the vendor table does not state: every tensor rank ≤ 4
  (checked after export, max rank 4 in all 15) and `nonconst_conv_args == 0`
  (`DEQUANTIZE == 0` everywhere).
- Numerics: each re-export is within the shipped FP16 model's own error floor against the FP32
  ONNX truth (`not_worse` per model; 5 realcugan-se models are bit-exact at 0.0, the rest differ by
  8.0e-4 … 5.5e-3, all below the shipped models' own 5.5e-3 floor). Against the shipped TFLite
  bytes the max diff is ≤ 5.535e-03, i.e. FP16 rounding, not a regression.
- APK packaging: `assets/litert-models` holds 15 models + `manifest.json`, and all 15 SHA-256
  values in the manifest match the bytes in the APK. Total model payload 70 056 512 B (the SPAN
  model grew by 368 B when the `Mul(255)` adapter was added).
- `:app:testDebugUnitTest` — 174 tests, 0 failures, 0 errors, 0 skipped.
- `:app:connectedDebugAndroidTest` on an API 35 x86_64 emulator — 4/4 MediaTek tests pass, including
  15/15 models executing through the LiteRT C ABI and the tiling path, and
  `w2xExNpuPageSkipsAssetExtraction` (0.009 s). The 9 failures in the same run are device-capability
  assumptions: 6 `QualcommHtpDeviceTest` cases need a Hexagon HTP, one needs a Depth Anything DLC
  asset, and two are `AssumptionViolatedException` from uninstalled extensions.
- Shipped APK (`upscale-app-arm64-v8a-debug.apk`, 290 049 105 B, sha256
  `3c26274366e57e7fac0a38b5a91130774714fe8499254f256a5de39c94454bf0`) contains 15 models whose
  SHA-256 all match `manifest.json`, `libwaifu2x-jni.so` with the `nativeInitBackendOnly` symbol, and
  the new Kotlin entry point in `classes23.dex`.
- The native plugin path is packaged: `lib/arm64-v8a/libLiteRtCompilerPlugin_MediaTek.so` and
  `lib/arm64-v8a/libLiteRtDispatch_MediaTek.so` are present in the arm64 APK.

### What the on-device log proved

A `mihon_crash_logs.txt` capture from the Dimensity 9400+ device (11 060 lines, two app runs)
settled what the desktop auditing could not. The load-bearing lines:

| Log | Line | Meaning |
| :--- | ---: | :--- |
| `[npu_registry.cc:30] NPU accelerator registered.` | 577 | accelerator and plugin wiring are fine |
| `[compiled_model.cc:750] NPU JIT compilation caching enabled with cache dir: …` | 586 | model file, manifest and cache dir open successfully |
| `[compiler_plugin.cc:742] Partitioned subgraph<0>, selected 124 ops, from a total of 127 ops.` | 962 | the graph reaches MediaTek's own compiler |
| `NEON Kernal only support at most rank 5` / `TransposeLayer … the rank of input tensor should be in range [0, 4]` | 942 / 948 | rejection #1, tensor rank |
| `MapTransposeConv2D: Weights as inputs not supported` / `NEON: dynamic weight is unsupported` | 2539 / 6385 | rejection #2, non-constant filter |
| `NeuronCompilation_finish failed with error 6` → `Compile model: 504` | 2046 → 2051 | final status, `kLiteRtStatusErrorCompilation` (turn 4, now fixed) |

A second capture, `mihon_crash_logs(1).txt` (13 381 lines), is the run **after** those fixes and is
what proves the NPU is genuinely executing:

| Log | Line | Meaning |
| :--- | ---: | :--- |
| `[npu_registry.cc:30] NPU accelerator registered.` | 3337, 3927, 5993, 6595, 7127, 7792, 8309, 10615, 11415, 11965 | accelerator up for every prepare |
| `[compiler_plugin.cc:742] Partitioned subgraph<0>, selected N op, from a total of N op. resulted in 1 partitions.` | 3611, 6856, 8036, 10874, 11654 | se 100/100, pro 98/98, animevideov3 97/97, w2xex 95/95, span 82/82 — one partition, nothing left on the host |
| `[litert_dispatch.cc:186] Loading shared library: …/libLiteRtDispatch_MediaTek.so` | 3868 | vendor dispatch loaded |
| `Ready: … (fully delegated NPU) layout=… crop=… downsample=… target=…` | 3900, 7115, 8292, 11126, 11889 | all five graphs fully delegated |
| `[gpu_registry.cc:160] GPU accelerator could not be loaded and registered.` | 3343 | registry probing only; no model fell back to GPU/CPU |

That capture covers five graphs (`realcugan-se-x2-no-denoise`, `realcugan-pro-x2-no-denoise`,
`realesrgan-animevideov3-x2`, `w2xex-photo-small-x2`, `span-nomosuni-x2`). The pro-3x trio,
`realesrgan-general-x4v3` and the remaining denoise variants do **not** appear in it; they share the
same rewrite and compile path but are not individually verified on device.

Both rejections are now fixed structurally (rank-4 chain, constant float32 weights), and the SELinux
lines in the same log are **not** part of the failure — see
`tools/downloads/selinux_evidence.md`: the 7 `avc: denied` entries name a display-feature prop, a
sysfs performance knob and a zygote socket, none of which is on the model or compile path, and the
43 `EX_SECURITY` `isDeclared` denials are followed every time by a successful
`[neuron_adapter_api.cc:158] Loaded NeuronAdapter shared library.`

**Verified on device** (from `mihon_crash_logs(1).txt`): NeuroPilot compiles these graphs,
`LiteRtCompiledModelIsFullyAccelerated` returns true, and every graph runs as a single fully delegated
partition.

**Not verified — requires a MediaTek device:**

- Whether the tile-seam feathering removes the visible line. The neighbour disagreement that causes it
  is measured and reduced, but the reported 1–3 row dip could not be reproduced offline.
- Whether the SPAN page now renders correctly end to end. The `Mul(255)` adapter is proven bit-exact
  against the old model fed `x255`, and the black output is proven gone offline, but no on-device
  screenshot exists yet.
- Whether W2xEX enhances an offline page on hardware. The asset-extraction dependency is removed and
  covered by an emulator test; the device scenario has not been rerun.
- Latency, perceived speed, and compiler-cache behaviour across app upgrades.
- The pro-3x trio, `realesrgan-general-x4v3`, and the denoise variants do not appear in either log. Google's
[MediaTek guide](https://developers.google.cn/edge/litert/next/mediatek) lists Dimensity 7300,
8300, 9000, 9200, 9300, 9400 and 9500; that is a documented platform list, not a compatibility
claim for every model, firmware or SoC revision.

### How to check it on hardware

1. Install `upscale-app-arm64-v8a-debug.apk`.
2. Settings → Reader → Image enhancement → backend **MediaTek NPU**, pick any model.
3. Watch the preparation status line. Success means compilation and full delegation both passed.
   `grep -c "fully delegated NPU"` should equal the number of models used.
4. Check the three correctness fixes:
   - Screenshot a `realcugan-se-x2-no-denoise` and a `realcugan-pro-2x-conservative` page at native
     size and profile the white margin rows (`tools/mtk_ops/seam_shape.py`); the seam dip should be
     well under 10 codes.
   - In airplane mode, select `w2xex-photo-small`: it must produce an enhanced page, and logcat must
     not contain `Failed to prepare model assets` or `SocketTimeoutException`.
   - Select `span-nomosuni`: the page must render normally, not black.
5. If it reports a failure, capture logcat:
   `adb logcat -s waifu2x:V LiteRt:* AndroidRuntime:E | grep -iE "mediatek|neuron|delegat|compil"`.
   `litert_backend::last_error()` carries the specific rejection reason — a
   "could not be fully delegated" message means the plugin compiled but kept ops on the host.

## References

- [MediaTek NPU with LiteRT](https://developers.google.cn/edge/litert/next/mediatek)
- [LiteRT NPU acceleration](https://developers.google.cn/edge/litert/next/npu) — on-device
  compilation, vendor runtime packaging, `CompilerCacheDir` persistence.
- Local copy of the compiler operator/data-type tables: `tools/downloads/mtk_compiler.md`.
