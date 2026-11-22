package eu.kanade.tachiyomi.util.mediatek

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Pins the MediaTek output-geometry dispatch against the declarations of the 15 models shipped in
 * `mediatek-models/`, so a re-exported model with a different output side is caught by a JVM test
 * instead of by a device that only logs "enhancement skipped".
 *
 * The rule under test is [MediaTekOutputLayout], the Kotlin mirror of `mediatek_layout::dispatch`
 * in `app/src/main/cpp/litert_backend.h`; the C++ side has its own host test
 * (`app/src/test/cpp/mediatek_layout_test.cpp`). Both encode this table:
 *
 * | model group                       | declared out    | layout | crop | down | target |
 * |-----------------------------------|-----------------|--------|------|------|--------|
 * | realcugan se/pro x2               | [1,3,512,512]   | NCHW   | 0    | 1    | 512    |
 * | realcugan pro x3                  | [1,3,792,792]   | NCHW   | 12   | 1    | 768    |
 * | realesrgan animevideov3 / general | [1,3,1024,1024] | NCHW   | 0    | 2    | 512    |
 * | span-nomosuni                     | [1,512,512,3]   | NHWC   | 0    | 1    | 512    |
 * | w2xex-photo-small                 | [1,3,512,512]   | NCHW   | 0    | 1    | 512    |
 */
class MediaTekModelLayoutTest {
    @Test
    fun `shipped model set is complete`() {
        val present = modelsDirectory().listFiles { file -> file.extension == "tflite" }.orEmpty()
            .map { it.nameWithoutExtension }
            .sorted()
        assertEquals(EXPECTED_MODELS.sorted(), present, "mediatek-models/ must contain exactly the 15 NPU models")
    }

    @Test
    fun `every shipped model declares 256 square input and resolves to a usable geometry`() {
        for (name in EXPECTED_MODELS) {
            val file = File(modelsDirectory(), "$name.tflite")
            assertTrue(file.isFile, "$name.tflite is missing")
            val shapes = TfliteShapes.read(file)
            assertEquals(listOf(1, 256, 256, 3), shapes.input, "$name input tensor")
            val scale = requestedScale(name)
            val plan = MediaTekOutputLayout.dispatch(shapes.output, scale)
            assertTrue(plan.ok, "$name ${shapes.output}: ${plan.error}")
            assertEquals(256 * scale, plan.targetSide, "$name target side")
            // float_tiles::process indexes out[(y*output_tile+x)*3+c] with output_tile = 256*scale
            // and rejects any other length, so this is the hard contract run() must satisfy.
            val emitted = plan.targetSide * plan.targetSide * 3
            assertEquals(256 * scale * 256 * scale * 3, emitted, "$name run() buffer length")
        }
    }

    @Test
    fun `realcugan x2 models keep the declared side and no reduction`() {
        val names = EXPECTED_MODELS.filter { it.startsWith("realcugan-") && "-x2-" in it }
        assertEquals(8, names.size)
        for (name in names) {
            val plan = dispatchOf(name)
            assertEquals(MediaTekOutputLayout.Layout.NCHW, plan.layout, name)
            assertEquals(512, plan.outputSide, name)
            assertEquals(0, plan.crop, name)
            assertEquals(1, plan.downsample, name)
            assertEquals(512, plan.targetSide, name)
        }
    }

    @Test
    fun `realcugan pro x3 trims the 12 pixel graph padding to 768`() {
        val names = EXPECTED_MODELS.filter { it.startsWith("realcugan-pro-x3-") }
        assertEquals(3, names.size)
        for (name in names) {
            val plan = dispatchOf(name)
            assertEquals(MediaTekOutputLayout.Layout.NCHW, plan.layout, name)
            assertEquals(792, plan.outputSide, name)
            assertEquals(12, plan.crop, name)
            assertEquals(1, plan.downsample, name)
            assertEquals(768, plan.targetSide, name)
            assertEquals(768, plan.outputSide - 2 * plan.crop, "$name trimmed side")
        }
    }

    @Test
    fun `4x native real-esrgan models are box reduced to the requested 2x page`() {
        for (name in listOf("realesrgan-animevideov3-x2", "realesrgan-general-x4v3-x2")) {
            val plan = dispatchOf(name)
            assertEquals(MediaTekOutputLayout.Layout.NCHW, plan.layout, name)
            assertEquals(1024, plan.outputSide, name)
            assertEquals(0, plan.crop, name)
            assertEquals(2, plan.downsample, name)
            assertEquals(512, plan.targetSide, name)
        }
    }

    /**
     * The geometry chain that could hide a silent scaling bug: the graph is 4x, the page is 2x.
     * `litert_backend::run()` must emit exactly `(256*runtime.scale)^2*3` floats because
     * `float_tiles::process` computes `output_tile = tile * scale` from that same `runtime.scale`
     * and rejects any other length. So `targetSide` must equal `256 * requestedScale`, and the
     * reduction must not shrink the page a second time.
     */
    @Test
    fun `downsampled run output matches the float_tiles output_tile contract`() {
        for (name in EXPECTED_MODELS) {
            val plan = dispatchOf(name)
            val scale = requestedScale(name)
            // float_tile_processor.h: const int output_tile = tile * scale;
            val outputTile = 256 * scale
            assertEquals(outputTile, plan.targetSide, "$name: run() target must be float_tiles' output_tile")
            assertEquals(
                outputTile * outputTile * 3,
                plan.targetSide * plan.targetSide * 3,
                "$name: run() must emit output_tile^2*3 floats",
            )
            // The declared tensor is consumed exactly: trimmed side == emitted side * reduction,
            // so no row or column is dropped or duplicated at the tail.
            assertEquals(
                plan.targetSide * plan.downsample,
                plan.outputSide - 2 * plan.crop,
                "$name: trimmed ${plan.outputSide - 2 * plan.crop} must equal ${plan.targetSide} * ${plan.downsample}",
            )
        }
        // The two 4x graphs and the 3x graph are the cases where those three numbers differ.
        val esr = dispatchOf("realesrgan-animevideov3-x2")
        assertEquals(1024, esr.targetSide * esr.downsample, "4x graph consumed completely")
        assertEquals(1024, esr.outputSide - 2 * esr.crop, "no border to trim on the 4x graphs")
        assertEquals(512, esr.targetSide, "2x page delivered")
        val pro3 = dispatchOf("realcugan-pro-x3-no-denoise")
        assertEquals(768, pro3.outputSide - 2 * pro3.crop, "3x graph trimmed to 768")
        assertEquals(1, pro3.downsample, "3x graph is not reduced")
    }

    @Test
    fun `span is the only NHWC model`() {
        val nhwc = EXPECTED_MODELS.filter { dispatchOf(it).layout == MediaTekOutputLayout.Layout.NHWC }
        assertEquals(listOf("span-nomosuni-x2"), nhwc)
        val plan = dispatchOf("span-nomosuni-x2")
        assertEquals(512, plan.outputSide)
        assertEquals(0, plan.crop)
        assertEquals(1, plan.downsample)
    }

    @Test
    fun `declared shape drives the decision not the model name`() {
        // A 3x-named model declaring 768 needs no crop; a graph declaring 792 does.
        val exact = MediaTekOutputLayout.dispatch(listOf(1, 3, 768, 768), 3)
        assertTrue(exact.ok, exact.error)
        assertEquals(0, exact.crop)
        assertEquals(768, exact.targetSide)
        val padded = MediaTekOutputLayout.dispatch(listOf(1, 3, 792, 792), 3)
        assertTrue(padded.ok, padded.error)
        assertEquals(12, padded.crop)
        assertEquals(768, padded.targetSide)
        // Reading the same 792 graph as 2x leaves 768 trimmed pixels for a 512 target, which is
        // not a whole reduction, so it must be refused rather than silently mis-scaled.
        val mismatched = MediaTekOutputLayout.dispatch(listOf(1, 3, 792, 792), 2)
        assertTrue(!mismatched.ok, "792 declared at 2x must be rejected")
        assertTrue(mismatched.error.isNotEmpty())
    }

    @Test
    fun `unexpected declarations fail loudly instead of guessing`() {
        val rejected = listOf(
            listOf(1, 3, 512, 512) to 4,
            listOf(1, 3, 512, 512) to 0,
            listOf(2, 3, 512, 512) to 2,
            listOf(3, 512, 512) to 2,
            listOf(1, 4, 512, 512) to 2,
            listOf(1, 1024, 1024, 4) to 2,
            listOf(1, 3, 768, 768) to 2,
            listOf(1, 3, 1024, 1024) to 3,
            listOf(1, 3, 256, 256) to 2,
            listOf(1, 3, 1536, 1536) to 2,
            listOf(1, 3, 512, 768) to 2,
            emptyList<Int>() to 2,
        )
        for ((shape, scale) in rejected) {
            val plan = MediaTekOutputLayout.dispatch(shape, scale)
            assertTrue(!plan.ok, "$shape at $scale must be rejected")
            assertTrue(plan.error.isNotEmpty(), "$shape at $scale needs an error message")
        }
    }

    private fun dispatchOf(name: String): MediaTekOutputLayout.Dispatch = MediaTekOutputLayout.dispatch(
        TfliteShapes.read(File(modelsDirectory(), "$name.tflite")).output,
        requestedScale(name),
    )

    /** Mirrors `MediaTekNpu.configure`: 3 only when the model name carries the x3 marker. */
    private fun requestedScale(name: String) = if ("-x3-" in name) 3 else 2

    private fun modelsDirectory(): File {
        val found = generateSequence(File("").absoluteFile) { it.parentFile }
            .map { File(it, MODELS_DIR) }
            .firstOrNull { it.isDirectory }
        assertTrue(found != null, "$MODELS_DIR/ not found above ${File("").absolutePath}")
        return requireNotNull(found)
    }

    private companion object {
        const val MODELS_DIR = "mediatek-models"

        /** Must match `qnnModelNames` in app/build.gradle.kts, which also names the staged assets. */
        val EXPECTED_MODELS = listOf(
            "realesrgan-animevideov3-x2",
            "realesrgan-general-x4v3-x2",
            "realcugan-se-x2-no-denoise",
            "realcugan-se-x2-denoise1x",
            "realcugan-se-x2-denoise2x",
            "realcugan-se-x2-denoise3x",
            "realcugan-se-x2-conservative",
            "realcugan-pro-x2-no-denoise",
            "realcugan-pro-x2-denoise3x",
            "realcugan-pro-x2-conservative",
            "realcugan-pro-x3-no-denoise",
            "realcugan-pro-x3-denoise3x",
            "realcugan-pro-x3-conservative",
            "span-nomosuni-x2",
            "w2xex-photo-small-x2",
        )
    }
}

/**
 * Kotlin mirror of `mediatek_layout::dispatch` in `app/src/main/cpp/litert_backend.h`. Pure, so the
 * device path and this test can be compared value by value without a MediaTek device.
 */
internal object MediaTekOutputLayout {
    enum class Layout { NHWC, NCHW }

    data class Dispatch(
        val ok: Boolean = false,
        val layout: Layout = Layout.NHWC,
        val outputSide: Int = 0,
        val crop: Int = 0,
        val downsample: Int = 1,
        val targetSide: Int = 0,
        val error: String = "",
    )

    private const val INPUT_SIDE = 256
    private const val MIN_NATIVE_SCALE = 2
    private const val MAX_NATIVE_SCALE = 4
    private const val MAXIMUM_PAD = 32
    private const val REQUIRED_RANK = 4

    fun dispatch(shape: List<Int>, requestedScale: Int): Dispatch {
        if (requestedScale != 2 && requestedScale != 3) {
            return Dispatch(error = "Unsupported model scale")
        }
        if (shape.size != REQUIRED_RANK) return Dispatch(error = "Expected a rank-4 output tensor")
        if (shape[0] != 1) return Dispatch(error = "Expected a single-image batch")

        val layout: Layout
        val side: Int
        if (shape[3] == 3 && shape[1] > 0 && shape[1] == shape[2]) {
            layout = Layout.NHWC
            side = shape[1]
        } else if (shape[1] == 3 && shape[2] > 0 && shape[2] == shape[3]) {
            layout = Layout.NCHW
            side = shape[2]
        } else {
            return Dispatch(error = "Unsupported output layout: expected NHWC [1,S,S,3] or NCHW [1,3,S,S]")
        }

        // Exact graphs first, then padded ones, so 768 at 3x is not also read as 2x with a 128 pad.
        var matches = 0
        var crop = 0
        var pass = 0
        while (pass < 2 && matches == 0) {
            for (candidate in MIN_NATIVE_SCALE..MAX_NATIVE_SCALE) {
                if (side % candidate != 0) continue
                val core = side / candidate
                if (core < INPUT_SIDE) continue
                val pad = core - INPUT_SIDE
                if (pad % 2 != 0 || pad / 2 > MAXIMUM_PAD) continue
                if ((pass == 0) != (pad == 0)) continue
                matches++
                crop = pad / 2 * candidate
            }
            pass++
        }
        if (matches == 0) {
            return Dispatch(
                layout = layout,
                outputSide = side,
                error = "Unsupported output side: expected 256*n scaled for 2x, 3x or 4x",
            )
        }
        if (matches > 1) return Dispatch(layout = layout, outputSide = side, error = "Ambiguous model output side")

        val target = INPUT_SIDE * requestedScale
        val effective = side - 2 * crop
        if (effective % target != 0) {
            return Dispatch(
                layout = layout,
                outputSide = side,
                error = "Model output side does not match the requested scale",
            )
        }
        val downsample = effective / target
        if (downsample != 1 && downsample != 2) {
            return Dispatch(layout = layout, outputSide = side, error = "Unsupported output reduction factor")
        }
        if (crop != 0 && downsample != 1) {
            return Dispatch(
                layout = layout,
                outputSide = side,
                error = "Padded output with an unexpected native scale",
            )
        }
        return Dispatch(true, layout, side, crop, downsample, target, "")
    }
}

/** Read-only TFLite flatbuffer reader for the declared shapes of subgraph 0's IO tensors. */
internal object TfliteShapes {
    data class Shapes(val input: List<Int>, val output: List<Int>)

    fun read(file: File): Shapes {
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val model = u32(buffer, 0).toInt()
        val subgraph = vectorTable(buffer, model, SUBGRAPHS, 0)
        val inputs = intVector(buffer, subgraph, 1)
        val outputs = intVector(buffer, subgraph, 2)
        require(inputs.isNotEmpty() && outputs.isNotEmpty()) { "${file.name} declares no IO tensors" }
        return Shapes(shape(buffer, subgraph, inputs[0]), shape(buffer, subgraph, outputs[0]))
    }

    private const val SUBGRAPHS = 2
    private const val TENSORS = 0
    private const val SHAPE = 0

    private fun shape(buffer: ByteBuffer, subgraph: Int, tensorIndex: Int): List<Int> =
        intVector(buffer, vectorTable(buffer, subgraph, TENSORS, tensorIndex), SHAPE)

    private fun fieldOffset(buffer: ByteBuffer, table: Int, field: Int): Int {
        val vtable = table - i32(buffer, table)
        val size = u16(buffer, vtable)
        val entry = 4 + 2 * field
        if (entry >= size) return -1
        val offset = u16(buffer, vtable + entry)
        return if (offset == 0) -1 else table + offset
    }

    private fun vectorTable(buffer: ByteBuffer, table: Int, field: Int, index: Int): Int {
        val pos = fieldOffset(buffer, table, field)
        require(pos >= 0) { "missing vector field $field" }
        val vector = pos + u32(buffer, pos).toInt()
        require(index < u32(buffer, vector).toInt()) { "vector index $index out of range" }
        val element = vector + 4 + 4 * index
        return element + u32(buffer, element).toInt()
    }

    private fun intVector(buffer: ByteBuffer, table: Int, field: Int): List<Int> {
        val pos = fieldOffset(buffer, table, field)
        if (pos < 0) return emptyList()
        val vector = pos + u32(buffer, pos).toInt()
        val length = u32(buffer, vector).toInt()
        return (0 until length).map { i32(buffer, vector + 4 + 4 * it) }
    }

    private fun u16(buffer: ByteBuffer, pos: Int) = buffer.getShort(pos).toInt() and 0xFFFF

    private fun i32(buffer: ByteBuffer, pos: Int) = buffer.getInt(pos)

    private fun u32(buffer: ByteBuffer, pos: Int) = buffer.getInt(pos).toLong() and 0xFFFFFFFFL
}
