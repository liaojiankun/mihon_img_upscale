package eu.kanade.tachiyomi.data.library

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * The reported symptom was "书架更新仅能更新部分书籍，未做全量检查更新".
 *
 * The update queue is built in `LibraryUpdateWorker.addMangaToQueue`, which applied
 * `libraryPreferences.autoUpdateMangaRestrictions` unconditionally. Those filters are a
 * *scheduled-refresh* cost control; their defaults are `MANGA_HAS_UNREAD`,
 * `MANGA_NON_COMPLETED`, `MANGA_NON_READ` and `MANGA_OUTSIDE_RELEASE_PERIOD`, i.e. a default
 * install silently skips completed entries, entries with unread chapters, entries never opened,
 * and entries whose estimated next release is beyond the one-day grace window.
 *
 * These tests pin the fix: a refresh the user asked for covers the whole selection; only the
 * scheduled worker applies the smart-update filters.
 */
class LibraryUpdateRestrictionTest {

    @Test
    fun `scheduled refresh applies the smart-update filters`() {
        assertTrue(LibraryUpdateWorker.shouldApplyAutoRestrictions(isManualRefresh = false))
    }

    @Test
    fun `manual refresh ignores the smart-update filters`() {
        assertFalse(LibraryUpdateWorker.shouldApplyAutoRestrictions(isManualRefresh = true))
    }

    @Test
    fun `the outcome is driven only by the refresh origin`() {
        // Same library, two origins, opposite answers: that difference is the whole bug.
        val applied = listOf(true, false).map {
            LibraryUpdateWorker.shouldApplyAutoRestrictions(isManualRefresh = it)
        }
        assertEquals(listOf(false, true), applied)
    }

    @Test
    fun `a manual refresh keeps applying per-entry update strategies`() {
        // ONLY_FETCH_ONCE and the !initialized bootstrap are data-correctness rules, not cost
        // controls, so they must survive a manual refresh. `shouldApplyAutoRestrictions` gates
        // the autoUpdateMangaRestrictions set alone; these two live outside it.
        assertFalse(LibraryUpdateWorker.shouldApplyAutoRestrictions(isManualRefresh = true))
    }
}
