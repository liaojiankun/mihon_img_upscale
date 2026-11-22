package eu.kanade.tachiyomi.util.mediatek

import android.content.Context
import android.os.Build
import android.util.Log
import eu.kanade.tachiyomi.BuildConfig
import eu.kanade.tachiyomi.util.npu.NpuModelStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** LiteRT compiles FP16 models on the user's MediaTek device and owns the compiler cache. */
object MediaTekNpu {
    const val CACHE_VERSION = "1"
    private const val RUNTIME_VERSION = "2.3.0"

    /**
     * The precisions MediaTek NeuroPilot can compile, per the vendor mapping in
     * `MediaTek_Neuro_Compiler.md` ("Supported Data Types"): Float32 / Float16 / Int32 / Int16 /
     * UInt8 / Int8 / Int4 / Int2 / Bool / Int64. There is **no Float8**, which is why the FP8
     * path was removed rather than kept as a dead gate.
     *
     * Of those, Int8 (`NEURON_TENSOR_QUANT8_SYMM` / `..._ASYMM_SIGNED`, per-tensor or per-channel)
     * is the narrowest that still works for convolution stacks, so it is the fastest supported
     * choice and the one that would line the MediaTek path up with the Qualcomm W8A8 path.
     *
     * The models shipped in `mediatek-models/` are *FP16-weighted with FP32 activations*
     * (`DEQUANTIZE` + `CONV_2D` in FLOAT32), so NeuroPilot currently compiles them as
     * `NEURON_TENSOR_FLOAT16` and Int8 cannot be selected for them. Enabling Int8 needs
     * INT8-quantized re-exports of the same 15 graphs — see docs/mediatek-npu.md.
     */
    const val PRECISION_FP16 = 0

    /** Int8 per-tensor/per-channel; supported by NeuroPilot, unused until INT8 exports exist. */
    const val PRECISION_INT8 = 2

    /** The precision the shipped MediaTek models are compiled at. */
    const val DEFAULT_PRECISION = PRECISION_FP16

    /** Precisions this build can actually serve, i.e. ones with matching exported models. */
    fun supportedPrecisions(): List<Int> = listOf(PRECISION_FP16)
    private val mutableStates = MutableStateFlow<Map<String, NpuModelStatus>>(emptyMap())
    val states = mutableStates.asStateFlow()

    init {
        runCatching { System.loadLibrary("waifu2x-jni") }
    }

    fun isDeviceSupported(): Boolean = Build.VERSION.SDK_INT >= 31 &&
        Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a" &&
        (
            Build.SOC_MANUFACTURER.contains("mediatek", ignoreCase = true) ||
                Build.HARDWARE.startsWith("mt", ignoreCase = true)
            )

    fun isActive(): Boolean = runCatching { nativeIsActive() }.getOrDefault(false)

    private fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private fun digest(file: File): String {
        val hash = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(65536)
            while (true) {
                val size = input.read(buffer)
                if (size < 0) break
                hash.update(buffer, 0, size)
            }
        }
        return hash.digest().joinToString("") { "%02x".format(it) }
    }

    /**
     * Resolves, verifies and stages the model file, then blocks while the single
     * Neuron compilation happens. Failed models are retried only by an explicit UI
     * request.
     */
    @Synchronized
    fun prepare(context: Context, modelName: String, retry: Boolean = false): File? =
        prepareInternal(context, modelName, retry, cpuValidationOnly = false)

    /** Exercises the same C ABI and tile implementation on test devices without an MTK NPU. */
    @Synchronized
    internal fun prepareForCpuValidation(context: Context, modelName: String): File? {
        check(BuildConfig.DEBUG)
        return prepareInternal(context, modelName, retry = true, cpuValidationOnly = true)
    }

    @Synchronized
    fun initialize(context: Context, modelName: String): Boolean {
        val model = prepare(context, modelName) ?: return false
        // The one and only compilation for this model in this process; prepare()
        // no longer compiles ahead of it.
        val error = runCatching {
            configure(context, modelName, model, cpuTest = false)
        }.getOrElse { it.message ?: "Unable to initialize MediaTek NPU" }
        if (error.isNotEmpty()) {
            mutableStates.update { it + (modelName to NpuModelStatus.Failed(error)) }
            return false
        }
        return isActive()
    }

    private fun configure(context: Context, name: String, file: File, cpuTest: Boolean): String {
        val scale = if ("-x3-" in name) 3 else 2
        val padding = when {
            // Real-CUGAN 2x must NOT use 18. Measured on device and in an exact
            // replay of float_tiles (tmp/r3_padtest.py, realcugan-se-x2-no-denoise,
            // 1058x1500 page, hard-copy stitching):
            //     pad 16 -> max seam step 0.0011
            //     pad 18 -> max seam step 0.2222   (one near-black row per 440)
            //     pad 20 -> max seam step 0.0014
            //     pad 24 -> max seam step 0.0018
            // core = 256 - 2*pad, so pad 20 gives core 216 and pad 18 gives 220.
            // At pad 18 the model's first core row lands exactly on a source row
            // that its own input no longer contains (the tile spans
            // [ty-18, ty+238), so core row 0 has only 18 rows of upward context
            // while the row above it had 219). 18 is also the only value that
            // produces this: 16/20/24 are all clean, so this is a resonance of the
            // 220-pixel core with the model's receptive field, not a general
            // "more padding is better" effect. 20 costs 216/220 = 1.8 % more tiles.
            name.startsWith("realcugan-") -> if (scale == 3) 14 else 20
            name.startsWith("realesrgan-general-") -> 32
            name.startsWith("realesrgan-") -> 16
            name.startsWith("span-") -> 24
            else -> 10
        }
        val cache = File(file.parentFile, "compiler-cache")
        check(cache.isDirectory || cache.mkdirs()) { "Unable to create compiler cache" }
        return nativeInitialize(
            file.absolutePath,
            cache.absolutePath,
            context.applicationInfo.nativeLibraryDir,
            scale,
            padding,
            cpuTest,
        )
    }

    private fun prepareInternal(
        context: Context,
        modelName: String,
        retry: Boolean,
        cpuValidationOnly: Boolean,
    ): File? {
        if (!cpuValidationOnly && mutableStates.value[modelName] is NpuModelStatus.Failed && !retry) return null
        return runCatching {
            check(cpuValidationOnly || isDeviceSupported()) { "This device has no supported MediaTek NPU" }
            val manifest = JSONObject(
                context.assets.open("litert-models/manifest.json").bufferedReader().use { it.readText() },
            )
            require(modelName.matches(Regex("[a-z0-9-]+")) && manifest.has(modelName)) { "Unknown MediaTek model" }
            val modelHash = manifest.getString(modelName)
            val identity = listOf(
                CACHE_VERSION,
                RUNTIME_VERSION,
                modelHash,
                Build.FINGERPRINT,
                Build.SUPPORTED_ABIS.first(),
            ).joinToString("|")
            val key = digest(identity.toByteArray()).take(32)
            val root = if (cpuValidationOnly) "litert-cpu-validation" else "mediatek-npu"
            val directory = File(context.noBackupFilesDir, "$root/$modelName/$key")
            check(directory.isDirectory || directory.mkdirs()) { "Unable to create model directory" }
            val model = File(directory, "model.tflite")
            if (!model.isFile || digest(model) != modelHash) {
                val pending = File(directory, "model.tflite.partial")
                try {
                    context.assets.open("litert-models/$modelName.tflite").use { input ->
                        pending.outputStream().use(input::copyTo)
                    }
                    check(digest(pending) == modelHash) { "MediaTek model checksum mismatch" }
                    check(!model.exists() || model.delete()) { "Unable to replace model" }
                    check(pending.renameTo(model)) { "Unable to commit model" }
                } finally {
                    pending.delete()
                }
            }
            if (!cpuValidationOnly && mutableStates.value[modelName] is NpuModelStatus.Ready) return@runCatching model
            if (!cpuValidationOnly) mutableStates.update { it + (modelName to NpuModelStatus.Preparing) }
            // Compilation happens once, in `initialize`, against the same model file
            // and compiler cache prepared here. It used to run here as well via a
            // throwaway Runtime, which on device produced two Neuron compilations
            // per model per process (analysis/timing.md §B3: 7 prepare passes +
            // 12 initialize passes, 1484 ms total, two `Ready:` lines per model).
            if (cpuValidationOnly) {
                val error = configure(context, modelName, model, cpuTest = true)
                check(error.isEmpty()) { error }
            }
            if (!cpuValidationOnly) mutableStates.update { it + (modelName to NpuModelStatus.Ready(model)) }
            directory.parentFile?.listFiles()?.filter {
                it.isDirectory && it != directory
            }?.forEach { it.deleteRecursively() }
            model
        }.onFailure { error ->
            Log.w("MediaTekNpu", "Unable to prepare $modelName; enhancement skipped", error)
            if (!cpuValidationOnly) {
                mutableStates.update { it + (modelName to NpuModelStatus.Failed(error.message.orEmpty())) }
            }
        }.getOrNull()
    }

    private external fun nativeInitialize(
        model: String,
        cache: String,
        libraries: String,
        scale: Int,
        padding: Int,
        cpuTest: Boolean,
    ): String
    private external fun nativeIsActive(): Boolean
}
