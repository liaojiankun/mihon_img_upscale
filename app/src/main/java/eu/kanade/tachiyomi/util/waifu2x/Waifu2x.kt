@file:Suppress("ktlint:standard:max-line-length")

package eu.kanade.tachiyomi.util.waifu2x

import android.content.Context
import android.graphics.Bitmap
import eu.kanade.tachiyomi.util.mediatek.MediaTekNpu
import eu.kanade.tachiyomi.util.qnn.QnnModelContexts
import eu.kanade.tachiyomi.util.qnn.QualcommHtp
import java.io.BufferedInputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/**
 * Waifu2x image upscaler using ncnn.
 * Provides 2x upscaling with denoising for manga images.
 */
object Waifu2x {

    const val PROCESSING_BACKEND_VULKAN = 0
    const val PROCESSING_BACKEND_QUALCOMM_NPU = 1
    const val PROCESSING_BACKEND_MEDIATEK_NPU = 2
    const val MODEL_REAL_ESRGAN_ANIME = 2
    const val MODEL_W2XEX_PHOTO_SMALL = 9
    const val MODEL_SPAN_NOMOSUNI_PHOTO = 19
    const val REAL_ESRGAN_STYLE_ANIME = 0
    const val REAL_ESRGAN_STYLE_PHOTO = 1

    // Bump when bundled model assets change so existing installations refresh their cache.
    private const val BUNDLED_MODEL_CACHE_VERSION = "17"
    internal const val QNN_CONTEXT_CACHE_VERSION = QnnModelContexts.CACHE_VERSION

    @Volatile private var isInitialized = false

    @Volatile private var isRealCuganInitialized = false

    @Volatile private var isRealEsrganInitialized = false

    @Volatile private var isNoseInitialized = false

    @Volatile private var isWaifu2xInitialized = false

    @Volatile private var isAnime4kInitialized = false

    @Volatile private var isW2xExInitialized = false

    init {
        try {
            System.loadLibrary("waifu2x-jni")
        } catch (e: UnsatisfiedLinkError) {
            // Native library not available
        }
    }

    fun init(context: Context, noiseLevel: Int = 2, scale: Int = 2): Boolean {
        if (isInitialized) return true

        return synchronized(this) {
            if (isInitialized) return true

            val modelDir = extractModelsToCache(context, "waifu2x-models")
            if (modelDir == null) {
                return false
            }

            isInitialized = nativeInit(modelDir, noiseLevel, scale, 0, false)
            if (isInitialized) {
                // Invalidate all other models
                isRealCuganInitialized = false
                isRealEsrganInitialized = false
                isNoseInitialized = false
                isWaifu2xInitialized = false // Wait, I am Waifu2x (generic)
                isAnime4kInitialized = false
                isW2xExInitialized = false
            }
            isInitialized
        }
    }

    /**
     * Process a bitmap image with Waifu2x upscaling.
     *
     * @param input Input bitmap (will not be modified)
     * @return Upscaled bitmap, or null if processing failed
     */
    fun process(input: Bitmap, id: Int = -1): Bitmap? {
        if (!isInitialized) return null

        // Ensure input is in ARGB_8888 format
        val argbBitmap = if (input.config != Bitmap.Config.ARGB_8888) {
            input.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            input
        }

        return nativeProcess(argbBitmap, id)
    }

    // Track current config to detect changes (excludes tileSleepMs since that doesn't require model reload)
    private data class RealCuganConfig(
        val noise: Int,
        val scale: Int,
        val isPro: Boolean,
        val precision: Int,
        val fp16Arithmetic: Boolean,
        val processingBackend: Int,
    )

    @Volatile private var lastRealCuganConfig: RealCuganConfig? = null

    fun initRealCugan(
        context: Context,
        noiseLevel: Int,
        scale: Int,
        isPro: Boolean = false,
        tileSleepMs: Int = 0,
        tileSize: Int = 128,
        precision: Int = 0,
        fp16Arithmetic: Boolean = false,
        processingBackend: Int = PROCESSING_BACKEND_VULKAN,
    ): Boolean {
        val effectiveNoiseLevel = if (isPro && noiseLevel !in setOf(0, 3, 4)) 3 else noiseLevel
        val model = if (isPro) 1 else 0
        val resolvedBackend = resolveProcessingBackend(processingBackend, model, scale)
        val resolvedPrecision = resolvePrecision(precision, resolvedBackend, model, scale)
        val newConfig = RealCuganConfig(
            effectiveNoiseLevel,
            scale,
            isPro,
            resolvedPrecision,
            fp16Arithmetic,
            resolvedBackend,
        )

        // Fast path: if already initialized with same config, just update performance params and return
        if (
            isRealCuganInitialized && lastRealCuganConfig == newConfig &&
            resolvedBackend == PROCESSING_BACKEND_VULKAN
        ) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
            return true
        }

        return synchronized(this) {
            val currentConfig = RealCuganConfig(
                effectiveNoiseLevel,
                scale,
                isPro,
                resolvedPrecision,
                fp16Arithmetic,
                resolvedBackend,
            )

            // Force reinit only if model parameters changed (not tileSleepMs)
            if (lastRealCuganConfig != currentConfig) {
                android.util.Log.d(
                    "Waifu2x",
                    "Config changed from $lastRealCuganConfig to $currentConfig, reinitializing...",
                )
                isRealCuganInitialized = false
            }

            if (isRealCuganInitialized) {
                nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
                return initializeBackend(
                    context,
                    resolvedBackend,
                    qnnModelName(model, scale, effectiveNoiseLevel),
                    realCuganPadding(scale),
                    resolvedPrecision,
                )
            }

            val assetPath = if (isPro) "realcugan-pro-models" else "realcugan-models"
            // An NPU page runs entirely on the NPU model: LiteRT holds its own
            // compiled graph, and the only thing nativeProcess needs is the page
            // geometry. Loading the ncnn model here built a full Vulkan pipeline
            // (gpu instance, shaders, .param/.bin) that this page never uses —
            // measured at 1.46-1.88 s per cold page, 85.8-96.7 % of the whole
            // pre-inference segment (logs/116, analysis/timing.md §B4). It also
            // overwrote g_realcugan_pro, silently disabling the Real-CUGAN Pro
            // texture guard and letting the tile-seam fix miss that model.
            if (resolvedBackend != PROCESSING_BACKEND_VULKAN) {
                if (!initBackendOnly(
                        context,
                        resolvedBackend,
                        qnnModelName(model, scale, effectiveNoiseLevel),
                        realCuganPadding(scale),
                        scale,
                        resolvedPrecision,
                        realcuganPro = isPro,
                    )
                ) {
                    return false
                }
                lastRealCuganConfig = currentConfig
                isRealCuganInitialized = true
                isInitialized = false
                isRealEsrganInitialized = false
                isNoseInitialized = false
                isWaifu2xInitialized = false
                isAnime4kInitialized = false
                isW2xExInitialized = false
                android.util.Log.d(
                    "Waifu2x",
                    "Initialized Real-CUGAN: isPro=$isPro, noise=$effectiveNoiseLevel, scale=$scale, " +
                        "tileSleepMs=$tileSleepMs, tileSize=$tileSize, precision=${currentConfig.precision}, " +
                        "backend=${backendName(currentConfig.processingBackend)}",
                )
                return isRealCuganInitialized
            }
            val modelDir = extractModelsToCache(context, assetPath)
            if (modelDir == null) {
                return false
            }

            isRealCuganInitialized = nativeInitRealCugan(
                modelDir,
                effectiveNoiseLevel,
                scale,
                tileSleepMs,
                ncnnPrecision(currentConfig.precision),
                currentConfig.fp16Arithmetic,
                isPro,
            )
            if (isRealCuganInitialized) {
                lastRealCuganConfig = currentConfig
                nativeUpdatePerformanceConfig(tileSleepMs, tileSize)

                // Invalidate all other models
                isInitialized = false
                isRealEsrganInitialized = false
                isNoseInitialized = false
                isWaifu2xInitialized = false
                isAnime4kInitialized = false
                isW2xExInitialized = false

                android.util.Log.d(
                    "Waifu2x",
                    "Initialized Real-CUGAN: isPro=$isPro, noise=$effectiveNoiseLevel, scale=$scale, " +
                        "tileSleepMs=$tileSleepMs, tileSize=$tileSize, precision=${currentConfig.precision}, " +
                        "backend=${backendName(currentConfig.processingBackend)}",
                )
            }
            isRealCuganInitialized &&
                initializeBackend(
                    context,
                    resolvedBackend,
                    qnnModelName(model, scale, effectiveNoiseLevel),
                    realCuganPadding(scale),
                    resolvedPrecision,
                )
        }
    }

    // Track Real-ESRGAN config
    private data class RealEsrganConfig(
        val style: Int,
        val scale: Int,
        val precision: Int,
        val fp16Arithmetic: Boolean,
        val processingBackend: Int,
    )
    private var lastRealEsrganConfig: RealEsrganConfig? = null

    fun initRealESRGAN(
        context: Context,
        scale: Int,
        style: Int = REAL_ESRGAN_STYLE_ANIME,
        tileSleepMs: Int = 0,
        tileSize: Int = 128,
        precision: Int = 0,
        fp16Arithmetic: Boolean = false,
        processingBackend: Int = PROCESSING_BACKEND_VULKAN,
    ): Boolean = synchronized(this) {
        val isPhotoStyle = style == REAL_ESRGAN_STYLE_PHOTO
        val outputScale = if (isPhotoStyle) 2 else scale
        val resolvedBackend = resolveProcessingBackend(processingBackend, MODEL_REAL_ESRGAN_ANIME, outputScale)
        val resolvedPrecision =
            resolvePrecision(precision, resolvedBackend, MODEL_REAL_ESRGAN_ANIME, outputScale, style)
        val config = RealEsrganConfig(style, outputScale, resolvedPrecision, fp16Arithmetic, resolvedBackend)
        // Force reinit if config changed
        if (lastRealEsrganConfig != config) {
            android.util.Log.d(
                "Waifu2x",
                "Real-ESRGAN config changed from $lastRealEsrganConfig to $config, reinitializing...",
            )
            isRealEsrganInitialized = false
        }

        if (isRealEsrganInitialized) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
            return initializeBackend(
                context,
                resolvedBackend,
                qnnModelName(MODEL_REAL_ESRGAN_ANIME, outputScale, 0, style),
                if (isPhotoStyle) 32 else 16,
                resolvedPrecision,
            )
        }

        val assetPath = if (isPhotoStyle) {
            "realesrgan-models/v3-general"
        } else {
            "realesrgan-models/v3-anime"
        }
        // Same reasoning as initRealCugan: an NPU page never touches ncnn, so
        // building its Vulkan pipeline is pure per-page latency.
        if (resolvedBackend != PROCESSING_BACKEND_VULKAN) {
            if (!initBackendOnly(
                    context,
                    resolvedBackend,
                    qnnModelName(MODEL_REAL_ESRGAN_ANIME, outputScale, 0, style),
                    if (isPhotoStyle) 32 else 16,
                    outputScale,
                    resolvedPrecision,
                )
            ) {
                return false
            }
            lastRealEsrganConfig = config
            isRealEsrganInitialized = true
            isInitialized = false
            isRealCuganInitialized = false
            isNoseInitialized = false
            isWaifu2xInitialized = false
            isAnime4kInitialized = false
            isW2xExInitialized = false
            android.util.Log.d(
                "Waifu2x",
                "Initialized Real-ESRGAN animevideov3: outputScale=$outputScale, tileSleepMs=$tileSleepMs, " +
                    "tileSize=$tileSize, precision=${config.precision}, backend=${backendName(
                        config.processingBackend,
                    )}",
            )
            return isRealEsrganInitialized
        }
        val modelDir = extractModelsToCache(context, assetPath)
        if (modelDir == null) {
            return false
        }

        val modelScale = if (isPhotoStyle) 4 else outputScale
        isRealEsrganInitialized = nativeInitRealESRGAN(
            modelDir,
            modelScale,
            outputScale,
            ncnnPrecision(config.precision),
            config.fp16Arithmetic,
        )
        if (isRealEsrganInitialized) {
            lastRealEsrganConfig = config
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)

            // Invalidate all other models
            isInitialized = false
            isRealCuganInitialized = false
            isNoseInitialized = false
            isWaifu2xInitialized = false
            isAnime4kInitialized = false
            isW2xExInitialized = false

            val variant = if (isPhotoStyle) "general-x4v3" else "animevideov3"
            android.util.Log.d(
                "Waifu2x",
                "Initialized Real-ESRGAN $variant: outputScale=$outputScale, tileSleepMs=$tileSleepMs, tileSize=$tileSize, precision=${config.precision}, backend=${backendName(
                    config.processingBackend,
                )}",
            )
        }
        isRealEsrganInitialized &&
            initializeBackend(
                context,
                resolvedBackend,
                qnnModelName(MODEL_REAL_ESRGAN_ANIME, outputScale, 0, style),
                if (isPhotoStyle) 32 else 16,
                resolvedPrecision,
            )
    }

    private data class GenericModelConfig(val precision: Int, val fp16Arithmetic: Boolean)
    private var lastNoseConfig: GenericModelConfig? = null

    fun initNose(context: Context, tileSleepMs: Int = 0, tileSize: Int = 128, precision: Int = 0, fp16Arithmetic: Boolean = false): Boolean = synchronized(
        this,
    ) {
        val config = GenericModelConfig(precision.coerceIn(0, 3), fp16Arithmetic)
        if (lastNoseConfig != config) {
            isNoseInitialized = false
        }
        if (isNoseInitialized) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
            return true
        }

        val modelDir = extractModelsToCache(context, "waifu2x-models-nose")
        if (modelDir == null) {
            return false
        }

        isNoseInitialized = nativeInitNose(modelDir, config.precision, config.fp16Arithmetic)
        if (isNoseInitialized) {
            lastNoseConfig = config
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)

            // Invalidate all other models
            isInitialized = false
            isRealCuganInitialized = false
            isRealEsrganInitialized = false
            isWaifu2xInitialized = false
            isAnime4kInitialized = false
            isW2xExInitialized = false

            android.util.Log.d(
                "Waifu2x",
                "Initialized Nose model, tileSleepMs=$tileSleepMs, tileSize=$tileSize, precision=${config.precision}",
            )
        }
        isNoseInitialized
    }

    // Track Waifu2x config
    private data class Waifu2xConfig(val noise: Int, val scale: Int, val precision: Int, val fp16Arithmetic: Boolean)
    private var lastWaifu2xConfig: Waifu2xConfig? = null

    fun initWaifu2x(context: Context, noise: Int, scale: Int, tileSleepMs: Int = 0, tileSize: Int = 128, precision: Int = 0, fp16Arithmetic: Boolean = false): Boolean = synchronized(
        this,
    ) {
        val newConfig = Waifu2xConfig(noise, scale, precision.coerceIn(0, 3), fp16Arithmetic)

        // Force reinit if config changed
        if (lastWaifu2xConfig != newConfig) {
            android.util.Log.d(
                "Waifu2x",
                "Waifu2x config changed from $lastWaifu2xConfig to $newConfig, reinitializing...",
            )
            isWaifu2xInitialized = false
        }

        if (isWaifu2xInitialized) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
            return true
        }

        val modelDir = extractModelsToCache(context, "waifu2x-models")
        if (modelDir == null) return false

        isWaifu2xInitialized = nativeInit(modelDir, noise, scale, newConfig.precision, newConfig.fp16Arithmetic)
        if (isWaifu2xInitialized) {
            lastWaifu2xConfig = newConfig
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)

            // Invalidate all other models
            isInitialized = false
            isRealCuganInitialized = false
            isRealEsrganInitialized = false
            isNoseInitialized = false
            isAnime4kInitialized = false
            isW2xExInitialized = false

            android.util.Log.d(
                "Waifu2x",
                "Initialized Waifu2x: noise=$noise, scale=$scale, tileSleepMs=$tileSleepMs, tileSize=$tileSize, precision=${newConfig.precision}",
            )
        }
        isWaifu2xInitialized
    }

    fun initWaifu2xUpconv7(context: Context, noise: Int, scale: Int, tileSleepMs: Int = 0, tileSize: Int = 128, precision: Int = 0, fp16Arithmetic: Boolean = false): Boolean = synchronized(
        this,
    ) {
        val newConfig = Waifu2xConfig(noise, scale, precision.coerceIn(0, 3), fp16Arithmetic)

        // Force reinit if config changed
        if (lastWaifu2xConfig != newConfig) {
            android.util.Log.d(
                "Waifu2x",
                "Waifu2x UpConv7 config changed from $lastWaifu2xConfig to $newConfig, reinitializing...",
            )
            isWaifu2xInitialized = false
        }

        if (isWaifu2xInitialized) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
            return true
        }

        val modelDir = extractModelsToCache(context, "waifu2x-models-upconv7")
        if (modelDir == null) return false

        isWaifu2xInitialized =
            nativeInitWaifu2xUpconv7(modelDir, noise, scale, newConfig.precision, newConfig.fp16Arithmetic)
        if (isWaifu2xInitialized) {
            lastWaifu2xConfig = newConfig
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)

            // Invalidate all other models
            isInitialized = false
            isRealCuganInitialized = false
            isRealEsrganInitialized = false
            isNoseInitialized = false
            isAnime4kInitialized = false
            isW2xExInitialized = false

            android.util.Log.d(
                "Waifu2x",
                "Initialized Waifu2x UpConv7: noise=$noise, scale=$scale, tileSleepMs=$tileSleepMs, tileSize=$tileSize, precision=${newConfig.precision}",
            )
        }
        isWaifu2xInitialized
    }

    private data class W2xExConfig(
        val model: Int,
        val scale: Int,
        val precision: Int,
        val fp16Arithmetic: Boolean,
        val processingBackend: Int,
    )
    private var lastW2xExConfig: W2xExConfig? = null

    private data class W2xExModel(
        val stem: String,
        val scale: Int,
        val assetPath: String,
        val padding: Int = 10,
    )

    private fun w2xExModel(
        stem: String,
        scale: Int,
        family: String,
        padding: Int = 10,
    ): W2xExModel {
        return W2xExModel(stem, scale, "$family/$stem", padding)
    }

    private fun w2xExModelFor(model: Int): W2xExModel? {
        return when (model) {
            6 -> w2xExModel("Universal-Fast-W2xEX", 2, "w2xex-esrgan")
            8 -> w2xExModel("Omni-MiniV2-W2xEX", 2, "w2xex-esrgan")
            MODEL_W2XEX_PHOTO_SMALL -> w2xExModel("Photo-Small-W2xEX", 2, "w2xex-esrgan")
            16 -> w2xExModel("animejanai-v2-ultra-compact-x2", 2, "animejanai-ncnn-vulkan")
            18 -> w2xExModel("2x-sudo-UltraCompact", 2, "sudo-ultracompact")
            MODEL_SPAN_NOMOSUNI_PHOTO ->
                w2xExModel("2x-NomosUni-SPAN-multijpg-ldl", 2, "span-nomosuni", padding = 24)
            else -> null
        }
    }

    fun w2xExScaleFor(model: Int): Int? = w2xExModelFor(model)?.scale

    fun isW2xExModel(model: Int): Boolean = w2xExModelFor(model) != null

    /**
     * Initialise a page for one of the NPU backends without loading an ncnn model.
     *
     * The NPU paths execute through [eu.kanade.tachiyomi.util.mediatek.MediaTekNpu] /
     * QNN, which hold their own compiled model, so the only thing native needs here
     * is the page geometry (scale, tile padding) and the `g_waifu2x` instance that
     * `nativeProcess` writes into. This keeps a missing or undownloadable ncnn model
     * from failing an NPU page.
     */
    private fun initBackendOnly(
        context: Context,
        backend: Int,
        modelName: String?,
        padding: Int,
        scale: Int,
        precision: Int,
        realcuganPro: Boolean = false,
    ): Boolean {
        val resolved = if (backend == PROCESSING_BACKEND_QUALCOMM_NPU) precision else 0
        if (!nativeInitBackendOnly(scale, padding, realcuganPro)) return false
        isInitialized = false
        isRealCuganInitialized = false
        isRealEsrganInitialized = false
        isNoseInitialized = false
        isWaifu2xInitialized = false
        isAnime4kInitialized = false
        isW2xExInitialized = false
        // Only claim the page once the NPU backend actually accepted the model, so a
        // failed backend leaves every process* entry point returning null (original
        // image) instead of running an uninitialised native instance.
        if (!initializeBackend(context, backend, modelName, padding, resolved)) return false
        lastW2xExConfig = W2xExConfig(-1, scale, resolved, false, backend)
        isW2xExInitialized = true
        android.util.Log.d(
            "Waifu2x",
            "Backend-only init: model=$modelName, scale=$scale, padding=$padding, " +
                "backend=${backendName(backend)}",
        )
        return true
    }

    fun initW2xEx(
        context: Context,
        model: Int,
        scale: Int = 2,
        tileSleepMs: Int = 0,
        tileSize: Int = 128,
        precision: Int = 0,
        fp16Arithmetic: Boolean = false,
        processingBackend: Int = PROCESSING_BACKEND_VULKAN,
    ): Boolean = synchronized(this) {
        val selectedModel = w2xExModelFor(model) ?: return false
        val effectiveScale = selectedModel.scale
        val modelStem = selectedModel.stem
        // Resolve the backend before touching the ncnn assets. The NPU keeps its
        // own compiled model, so it neither needs the .param/.bin files nor may
        // be aborted by their download: that timeout is what made an offline
        // W2xEX page fall back to the original image (mihon_crash_logs(1).txt:11223).
        val resolvedBackend = resolveProcessingBackend(processingBackend, model, effectiveScale)
        val resolvedPrecision = resolvePrecision(precision, resolvedBackend, model, effectiveScale)
        if (resolvedBackend != PROCESSING_BACKEND_VULKAN) {
            return initBackendOnly(
                context,
                resolvedBackend,
                qnnModelName(model, effectiveScale, 0),
                selectedModel.padding,
                effectiveScale,
                resolvedPrecision,
            )
        }
        val modelDir = extractModelsToCache(context, selectedModel.assetPath) ?: return false

        val config = W2xExConfig(model, effectiveScale, resolvedPrecision, fp16Arithmetic, resolvedBackend)
        if (lastW2xExConfig != config) {
            isW2xExInitialized = false
        }

        if (isW2xExInitialized) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
            return initializeBackend(
                context,
                resolvedBackend,
                qnnModelName(model, effectiveScale, 0),
                selectedModel.padding,
                resolvedPrecision,
            )
        }

        isW2xExInitialized = nativeInitW2xEx(
            modelDir,
            modelStem,
            effectiveScale,
            ncnnPrecision(config.precision),
            config.fp16Arithmetic,
            selectedModel.padding,
        )
        if (isW2xExInitialized) {
            lastW2xExConfig = config
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)

            isInitialized = false
            isRealCuganInitialized = false
            isRealEsrganInitialized = false
            isNoseInitialized = false
            isWaifu2xInitialized = false
            isAnime4kInitialized = false

            android.util.Log.d(
                "Waifu2x",
                "Initialized generic ncnn model: $modelStem, scale=$effectiveScale, precision=${config.precision}, backend=${backendName(
                    config.processingBackend,
                )}",
            )
        }
        isW2xExInitialized &&
            initializeBackend(
                context,
                resolvedBackend,
                qnnModelName(model, effectiveScale, 0),
                selectedModel.padding,
                resolvedPrecision,
            )
    }

    // Reuse processRealCugan for all generic ncnn models
    // But check specific flags
    // Reuse processRealCugan for all generic ncnn models
    // But check specific flags
    fun processRealESRGAN(input: Bitmap, id: Int = -1): Bitmap? {
        if (!isRealEsrganInitialized) return null
        return processBitmapHelper(input, id)
    }

    fun processNose(input: Bitmap, id: Int = -1): Bitmap? {
        if (!isNoseInitialized) return null
        return processBitmapHelper(input, id)
    }

    fun processWaifu2x(input: Bitmap, id: Int = -1): Bitmap? {
        if (!isWaifu2xInitialized) return null
        return processBitmapHelper(input, id)
    }

    fun processW2xEx(input: Bitmap, id: Int = -1): Bitmap? {
        if (!isW2xExInitialized) return null
        return processBitmapHelper(input, id)
    }

    @Volatile var processingId: Int = -1

    private fun processBitmapHelper(input: Bitmap, id: Int): Bitmap? {
        if (input.isRecycled) return null

        val argbBitmap = if (input.config != Bitmap.Config.ARGB_8888) {
            try {
                input.copy(Bitmap.Config.ARGB_8888, false)
            } catch (e: Exception) {
                null
            }
        } else {
            input
        } ?: return null

        processingId = id
        try {
            val result = nativeProcessRealCugan(argbBitmap, id)
            return if (result === argbBitmap) null else result
        } finally {
            processingId = -1
            if (argbBitmap !== input) {
                argbBitmap.recycle()
            }
        }
    }

    /**
     * Get the raw packed progress value from native code.
     * Format: [ID (upper 32 bits)] [Progress (lower 32 bits)]
     */
    fun getProgress(): Long = nativeGetProgress()

    /**
     * Get only the progress percentage (0-100) from the packed value.
     */
    fun getProgressPercent(): Int {
        val packed = nativeGetProgress()
        return (packed and 0xFFFFFFFF).toInt()
    }

    /**
     * Get only the processing ID from the packed value.
     */
    fun getProgressId(): Int {
        val packed = nativeGetProgress()
        return (packed shr 32).toInt()
    }

    /**
     * Reset Real-CUGAN to allow re-initialization with new settings.
     */
    @Synchronized
    fun resetRealCugan() {
        isInitialized = false
        isRealCuganInitialized = false
        isRealEsrganInitialized = false
        isNoseInitialized = false
        isWaifu2xInitialized = false
        isAnime4kInitialized = false
        isW2xExInitialized = false
        lastRealCuganConfig = null
        lastRealEsrganConfig = null
        lastNoseConfig = null
        lastWaifu2xConfig = null
        lastW2xExConfig = null
        nativeDestroy()
    }

    /**
     * Ask any active native upscaling operation to stop at its next cancellation check.
     */
    fun abortProcessing() {
        nativeAbortProcessing()
    }

    fun prepareProcessing() {
        nativeClearAbortProcessing()
    }

    /**
     * Process bitmap with Real-CUGAN.
     */
    fun processRealCugan(input: Bitmap, id: Int = -1): Bitmap? {
        if (!isRealCuganInitialized) return null
        return processBitmapHelper(input, id)
    }

    /**
     * Release native resources.
     */
    fun destroy() {
        if (isInitialized || isRealCuganInitialized || isRealEsrganInitialized || isNoseInitialized ||
            isWaifu2xInitialized ||
            isAnime4kInitialized ||
            isW2xExInitialized
        ) {
            nativeDestroy()
            isInitialized = false
            isRealCuganInitialized = false
            isRealEsrganInitialized = false
            isNoseInitialized = false
            isWaifu2xInitialized = false
            isAnime4kInitialized = false
            isW2xExInitialized = false
        }
    }

    /**
     * Initialize Anime4K with specific mode.
     */
    fun initAnime4K(context: Context, mode: Int): Boolean {
        if (isAnime4kInitialized) return true

        val assetManager = context.assets
        val shaders = mutableListOf<String>()
        val names = mutableListOf<String>()

        fun addShader(name: String) {
            val content = assetManager.open("anime4k/$name").bufferedReader().use { it.readText() }
            shaders.add(content)
            names.add(name)
        }

        try {
            addShader("Anime4K_Clamp_Highlights.glsl")
            when (mode) {
                0 -> addShader("Anime4K_Restore_CNN_M.glsl") // Fast
                1 -> addShader("Anime4K_Restore_CNN_VL.glsl") // High
                2 -> { // Ultra
                    addShader("Anime4K_Restore_CNN_VL.glsl")
                    addShader("Anime4K_Upscale_CNN_x2_VL.glsl")
                }
            }
        } catch (e: Exception) {
            return false
        }

        isAnime4kInitialized = nativeInitAnime4K(shaders.toTypedArray(), names.toTypedArray())
        // Invalidate all other models
        if (isAnime4kInitialized) {
            isInitialized = false
            isRealCuganInitialized = false
            isRealEsrganInitialized = false
            isNoseInitialized = false
            isWaifu2xInitialized = false
            isW2xExInitialized = false
        }
        return isAnime4kInitialized
    }

    /**
     * Process bitmap with Anime4K.
     */
    fun processAnime4K(input: Bitmap): Bitmap? {
        if (!isAnime4kInitialized || input.isRecycled) return null

        val argbBitmap = try {
            if (input.config != Bitmap.Config.ARGB_8888) {
                input.copy(Bitmap.Config.ARGB_8888, true)
            } else {
                input.copy(Bitmap.Config.ARGB_8888, true) // Must be mutable for in-place
            }
        } catch (e: Exception) {
            null
        } ?: return null

        try {
            return nativeProcessAnime4K(argbBitmap)
        } finally {
            // We don't recycle argbBitmap if it's the same as input,
            // but here it's always a copy (true).
            // Actually, nativeProcessAnime4K returns the SAME bitmap (in-place)
            // so we SHOULD NOT recycle it here if it's the result.
        }
    }

    private fun extractModelsToCache(context: Context, assetPath: String): String? {
        return try {
            val cacheDir = File(context.cacheDir, assetPath)
            if (!cacheDir.exists()) {
                cacheDir.mkdirs()
            }

            val assetManager = context.assets
            val modelFiles = assetManager.list(assetPath).orEmpty()
            val assetVersionFile = File(cacheDir, ".bundled-model-version")
            val refreshBundledModels = modelFiles.isNotEmpty() &&
                assetVersionFile.takeIf(File::exists)?.readText() != BUNDLED_MODEL_CACHE_VERSION

            for (filename in modelFiles) {
                val outFile = File(cacheDir, filename)
                if (refreshBundledModels || !outFile.exists()) {
                    assetManager.open("$assetPath/$filename").use { input ->
                        outFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
            if (refreshBundledModels) {
                assetVersionFile.writeText(BUNDLED_MODEL_CACHE_VERSION)
            }

            if (!cacheDir.hasNcnnModels()) {
                downloadReleaseModels(assetPath, cacheDir)
            }

            if (!cacheDir.hasNcnnModels()) {
                downloadDirectModels(assetPath, cacheDir)
            }

            if (!cacheDir.hasNcnnModels()) {
                return null
            }

            cacheDir.absolutePath
        } catch (e: Exception) {
            android.util.Log.e("Waifu2x", "Failed to prepare model assets: $assetPath", e)
            null
        }
    }

    private data class ModelReleaseSource(
        val url: String,
        val entryPrefix: String,
        val stripPrefix: String = "",
    )

    private data class DirectModelSource(
        val baseUrl: String,
        val files: List<String>,
    )

    private fun releaseSourceFor(assetPath: String): ModelReleaseSource? {
        return when (assetPath) {
            "realcugan-models" -> ModelReleaseSource(
                url = "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip",
                entryPrefix = "realcugan-ncnn-vulkan-20220728-ubuntu/models-se/",
            )
            "realcugan-pro-models" -> ModelReleaseSource(
                url = "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip",
                entryPrefix = "realcugan-ncnn-vulkan-20220728-ubuntu/models-pro/",
            )
            "waifu2x-models-nose" -> ModelReleaseSource(
                url = "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip",
                entryPrefix = "realcugan-ncnn-vulkan-20220728-ubuntu/models-nose/",
            )
            "waifu2x-models" -> ModelReleaseSource(
                url = "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20250915/waifu2x-ncnn-vulkan-20250915-linux.zip",
                entryPrefix = "waifu2x-ncnn-vulkan-20250915-linux/models-cunet/",
            )
            "waifu2x-models-upconv7" -> ModelReleaseSource(
                url = "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20250915/waifu2x-ncnn-vulkan-20250915-linux.zip",
                entryPrefix = "waifu2x-ncnn-vulkan-20250915-linux/models-upconv_7_anime_style_art_rgb/",
            )
            "realesrgan-models/v3-anime" -> ModelReleaseSource(
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
                entryPrefix = "models/realesr-animevideov3-",
                stripPrefix = "realesr-animevideov3-",
            )
            else -> null
        }
    }

    private fun directSourceFor(assetPath: String): DirectModelSource? {
        val w2xExStem = assetPath.removePrefix("w2xex-esrgan/").takeIf { it != assetPath }
        if (w2xExStem != null) {
            val supported = setOf(
                "Universal-Fast-W2xEX",
                "Omni-MiniV2-W2xEX",
                "Photo-Small-W2xEX",
            )
            if (w2xExStem !in supported) return null

            return DirectModelSource(
                baseUrl = "https://huggingface.co/randomblock1/W2xEX-ESRGAN/resolve/main",
                files = listOf("$w2xExStem.param", "$w2xExStem.bin"),
            )
        }

        val animeJaNaiStem = assetPath.removePrefix("animejanai-ncnn-vulkan/").takeIf { it != assetPath }
        if (animeJaNaiStem == "animejanai-v2-ultra-compact-x2") {
            return DirectModelSource(
                baseUrl = "https://raw.githubusercontent.com/Justin62628/animejanai-ncnn-vulkan/main/models/$animeJaNaiStem",
                files = listOf("$animeJaNaiStem.param", "$animeJaNaiStem.bin"),
            )
        }

        return null
    }

    private fun downloadReleaseModels(assetPath: String, cacheDir: File) {
        val source = releaseSourceFor(assetPath) ?: return
        val connection = (URL(source.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        }
        try {
            ZipInputStream(BufferedInputStream(connection.inputStream)).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory || !entry.name.startsWith(source.entryPrefix)) {
                        zip.closeEntry()
                        continue
                    }
                    val rawName = entry.name.substringAfterLast('/')
                    if (!rawName.endsWith(".param") && !rawName.endsWith(".bin")) {
                        zip.closeEntry()
                        continue
                    }
                    val filename = rawName.removePrefix(source.stripPrefix)
                    val outFile = File(cacheDir, filename)
                    outFile.outputStream().use { output ->
                        zip.copyTo(output)
                    }
                    zip.closeEntry()
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun downloadDirectModels(assetPath: String, cacheDir: File) {
        val source = directSourceFor(assetPath) ?: return
        source.files.forEach { filename ->
            val outFile = File(cacheDir, filename)
            if (outFile.exists() && outFile.length() > 0L) return@forEach

            val tempFile = File(cacheDir, "$filename.tmp")
            val url = "${source.baseUrl}/$filename"
            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 120_000
                instanceFollowRedirects = true
            }
            try {
                BufferedInputStream(connection.inputStream).use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                if (!tempFile.renameTo(outFile)) {
                    tempFile.delete()
                }
            } finally {
                connection.disconnect()
            }
        }
    }

    private fun File.hasNcnnModels(): Boolean {
        val files = listFiles().orEmpty()
        return files.any { it.extension == "param" } && files.any { it.extension == "bin" }
    }

    fun setUiBusy(busy: Boolean) {
        nativeSetUiBusy(busy)
    }

    fun isQnnRuntimeAvailable(): Boolean = try {
        nativeIsQnnRuntimeAvailable()
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    private val deviceDefaultBackend by lazy {
        when {
            MediaTekNpu.isDeviceSupported() -> PROCESSING_BACKEND_MEDIATEK_NPU
            isQualcommNpuAvailable() -> PROCESSING_BACKEND_QUALCOMM_NPU
            else -> PROCESSING_BACKEND_VULKAN
        }
    }

    fun defaultProcessingBackend(): Int = deviceDefaultBackend

    fun isQualcommNpuAvailable(): Boolean = QualcommHtp.architecture() != null && isQnnRuntimeAvailable()

    fun isQualcommNpuModelSupported(model: Int, scale: Int): Boolean {
        if (model == 1) return scale == 2 || scale == 3
        if (scale != 2) return false
        return model == 0 ||
            model == MODEL_REAL_ESRGAN_ANIME ||
            model == MODEL_W2XEX_PHOTO_SMALL ||
            model == MODEL_SPAN_NOMOSUNI_PHOTO
    }

    fun qnnModelName(model: Int, scale: Int, noise: Int, realEsrganStyle: Int = REAL_ESRGAN_STYLE_ANIME): String? {
        if (!isQualcommNpuModelSupported(model, scale)) return null
        return when (model) {
            0, 1 -> {
                val effectiveNoise = if (model == 1 && noise !in setOf(0, 3, 4)) 3 else noise
                val variant = when (effectiveNoise) {
                    1 -> "denoise1x"
                    2 -> "denoise2x"
                    3 -> "denoise3x"
                    4 -> "conservative"
                    else -> "no-denoise"
                }
                "realcugan-${if (model == 1) "pro" else "se"}-x$scale-$variant"
            }
            MODEL_REAL_ESRGAN_ANIME -> if (realEsrganStyle == REAL_ESRGAN_STYLE_PHOTO) {
                "realesrgan-general-x4v3-x2"
            } else {
                "realesrgan-animevideov3-x2"
            }
            MODEL_W2XEX_PHOTO_SMALL -> "w2xex-photo-small-x2"
            MODEL_SPAN_NOMOSUNI_PHOTO -> "span-nomosuni-x2"
            else -> null
        }
    }

    // Preserve an explicit NPU choice even if the device/model cannot execute it.
    // Initialization then fails and the reader keeps the original image.
    fun resolveProcessingBackend(requestedBackend: Int, model: Int, scale: Int): Int =
        requestedBackend.takeIf { it in PROCESSING_BACKEND_VULKAN..PROCESSING_BACKEND_MEDIATEK_NPU }
            ?: PROCESSING_BACKEND_VULKAN

    fun resolvePrecision(
        requestedPrecision: Int,
        processingBackend: Int,
        model: Int,
        scale: Int,
        realEsrganStyle: Int = REAL_ESRGAN_STYLE_ANIME,
    ): Int {
        if (processingBackend == PROCESSING_BACKEND_QUALCOMM_NPU) {
            // Migrate the retired local W8A16 setting (4) only for an INT8-capable model.
            return if ((requestedPrecision == QnnModelContexts.PRECISION_W8A8 || requestedPrecision == 4) &&
                QnnModelContexts.supportsW8a8(qnnModelName(model, scale, 0, realEsrganStyle))
            ) {
                QnnModelContexts.PRECISION_W8A8
            } else {
                0
            }
        }
        if (resolveProcessingBackend(processingBackend, model, scale) != PROCESSING_BACKEND_VULKAN) return 0
        return requestedPrecision.takeIf { it == 0 || it == 1 || it == 3 } ?: 0
    }

    /**
     * Tile overlap for Real-CUGAN, shared by the ncnn path and the NPU
     * geometry-only init so the two can never diverge.
     *
     * 2x uses 20, not 18. Every tile is 256 px wide with core = 256 - 2*padding,
     * so padding decides which source rows land on the tile's first and last
     * output rows. Measured by replaying float_tiles exactly against the real
     * model on a 1058x1500 page (tmp/r3_padtest.py, realcugan-se-x2-no-denoise,
     * hard-copy stitching):
     *
     *     pad 16 -> max seam step 0.0011
     *     pad 18 -> max seam step 0.2222   <- one near-black row every 440
     *     pad 20 -> max seam step 0.0014
     *     pad 24 -> max seam step 0.0018
     *
     * On device the same 18 produced dark rows at phase 42 mod 440 in 5 of 7
     * tile seams of `realcugan-se-x2-no-denoise`, while Real-ESRGAN (16),
     * W2xEX (10) and SPAN (24) had 0 of 7. Only 18 resonates with the model;
     * 16/20/24 are all clean, so this is not "more padding is better".
     * 20 costs 216/220 = 1.8 % more tiles than 18.
     */
    private fun realCuganPadding(scale: Int): Int = if (scale == 3) 14 else 20

    private fun ncnnPrecision(precision: Int): Int = if (precision == QnnModelContexts.PRECISION_W8A8) 0 else precision

    private fun backendName(backend: Int): String = when (backend) {
        PROCESSING_BACKEND_QUALCOMM_NPU -> "Qualcomm NPU"
        PROCESSING_BACKEND_MEDIATEK_NPU -> "MediaTek NPU"
        else -> "Vulkan"
    }

    private fun initializeBackend(context: Context, backend: Int, modelName: String?, padding: Int, precision: Int): Boolean {
        nativeSetProcessingBackend(3) // Block execution until the selected model is ready.
        val ready = runCatching {
            when (backend) {
                PROCESSING_BACKEND_VULKAN -> true
                PROCESSING_BACKEND_QUALCOMM_NPU ->
                    modelName != null &&
                        initializeQnnIfAvailable(context, modelName, padding, precision)
                PROCESSING_BACKEND_MEDIATEK_NPU -> modelName != null && MediaTekNpu.initialize(context, modelName)
                else -> false
            }
        }.getOrDefault(false)
        nativeSetProcessingBackend(if (ready) backend else 3)
        return ready
    }

    fun isQnnActive(): Boolean = try {
        nativeIsQnnInitialized()
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    private fun initializeQnnIfAvailable(context: Context, modelName: String, padding: Int, precision: Int): Boolean {
        if (!isQualcommNpuAvailable()) return false
        if (isQnnActive()) return true
        return runCatching {
            val output = QnnModelContexts.prepare(context, modelName, precision = precision) ?: return false
            nativeInitQnn(output.absolutePath, context.applicationInfo.nativeLibraryDir, padding)
        }.onFailure {
            android.util.Log.w("Waifu2x", "Unable to initialize Qualcomm NPU; enhancement skipped", it)
        }.getOrDefault(false)
    }

    fun scaleBitmapNative(input: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap? {
        if (input.isRecycled) return null
        if (input.width == targetWidth && input.height == targetHeight) return input

        val argbBitmap = if (input.config != Bitmap.Config.ARGB_8888) {
            try {
                input.copy(Bitmap.Config.ARGB_8888, false)
            } catch (e: Exception) {
                null
            }
        } else {
            input
        } ?: return null

        return try {
            nativeScaleBitmap(argbBitmap, targetWidth, targetHeight)
        } finally {
            if (argbBitmap !== input) {
                argbBitmap.recycle()
            }
        }
    }

    // Native methods
    private external fun nativeInit(
        modelDir: String,
        noiseLevel: Int,
        scale: Int,
        precision: Int,
        fp16Arithmetic: Boolean,
    ): Boolean
    private external fun nativeInitWaifu2xUpconv7(
        modelDir: String,
        noiseLevel: Int,
        scale: Int,
        precision: Int,
        fp16Arithmetic: Boolean,
    ): Boolean
    private external fun nativeInitW2xEx(
        modelDir: String,
        modelStem: String,
        scale: Int,
        precision: Int,
        fp16Arithmetic: Boolean,
        padding: Int,
    ): Boolean
    private external fun nativeProcess(input: Bitmap, id: Int): Bitmap?
    private external fun nativeInitBackendOnly(scale: Int, padding: Int, realcuganPro: Boolean): Boolean
    private external fun nativeDestroy()
    private external fun nativeAbortProcessing()
    private external fun nativeClearAbortProcessing()
    private external fun nativeSetUiBusy(busy: Boolean)
    private external fun nativeSetProcessingBackend(backend: Int)
    private external fun nativeIsQnnRuntimeAvailable(): Boolean
    private external fun nativeInitQnn(contextPath: String, nativeLibraryDir: String, padding: Int): Boolean
    private external fun nativeIsQnnInitialized(): Boolean

    // ... (Anime4K signatures unchanged)

    private external fun nativeInitAnime4K(shaders: Array<String>, names: Array<String>): Boolean
    private external fun nativeProcessAnime4K(input: Bitmap): Bitmap?

    private external fun nativeInitRealCugan(modelDir: String, noiseLevel: Int, scale: Int, tileSleepMs: Int, precision: Int, fp16Arithmetic: Boolean, isPro: Boolean): Boolean
    private external fun nativeUpdatePerformanceConfig(tileSleepMs: Int, tileSize: Int)

    fun updatePerformance(tileSleepMs: Int, tileSize: Int) {
        if (isRealCuganInitialized || isRealEsrganInitialized || isNoseInitialized || isWaifu2xInitialized ||
            isW2xExInitialized
        ) {
            nativeUpdatePerformanceConfig(tileSleepMs, tileSize)
        }
    }

    private external fun nativeInitRealESRGAN(modelDir: String, modelScale: Int, outputScale: Int, precision: Int, fp16Arithmetic: Boolean): Boolean
    private external fun nativeInitNose(modelDir: String, precision: Int, fp16Arithmetic: Boolean): Boolean
    private external fun nativeProcessRealCugan(input: Bitmap, id: Int): Bitmap?
    private external fun nativeScaleBitmap(input: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap?
    private external fun nativeGetProgress(): Long
}
