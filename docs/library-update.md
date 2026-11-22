# Library update: why a refresh covered only part of the library

## Symptom

A manual "update library" (toolbar refresh or the global update action) refreshed only some of the
entries in the selected category. Entries that were already complete, had unread chapters, had never
been opened, or were not expected to release a chapter that day were silently left out. There was no
error in the UI and nothing in `failedUpdates` — those entries never entered the queue at all.

## Cause

`LibraryUpdateWorker.addMangaToQueue()` built the update queue and unconditionally applied
`libraryPreferences.autoUpdateMangaRestrictions`:

```kotlin
val restrictions = libraryPreferences.autoUpdateMangaRestrictions.get()
```

Those preferences are a cost control for the **scheduled** refresh. Their defaults
(`domain/src/main/java/eu/kanade/tachiyomi/domain/library/model/LibraryPreferences.kt:52-60`) are:

| Restriction | Default | Entry is skipped when |
| :--- | :---: | :--- |
| `MANGA_HAS_UNREAD` | on | the entry has unread chapters |
| `MANGA_NON_COMPLETED` | on | the entry's status is Completed |
| `MANGA_NON_READ` | on | the entry has no chapter marked read |
| `MANGA_OUTSIDE_RELEASE_PERIOD` | on | `nextUpdate > fetchWindowUpperBound` |

The fourth one accounts for most of the missing entries. `fetchInterval.getWindow()` returns
`(today - 1 day) .. (today + 1 day)` (`FetchInterval.kt:64-67`, `GRACE_PERIOD = 1`), while
`ReleaseSchedule.nextCheck()` / `backOff()` push `nextUpdate` as far as `MAX_INTERVAL = 28` days out.
Any entry whose last chapter released recently therefore sits outside the window and is filtered out
of a refresh the user explicitly triggered.

Both refresh origins reach the same code: `LibraryTab.kt:96-108` sends the toolbar refresh **and**
the global update through `LibraryUpdateWorker.startNow(...)`, which enqueues with
`WORK_NAME_MANUAL`. `doWork()` passed nothing to distinguish it, so a manual refresh was filtered
exactly like a scheduled one.

This is unrelated to the `WebViewInterceptor` change that restored `"origin"` to
`unsafeHeaderNames`. That fix addressed Cloudflare challenge failures, which surface as entries in
`failedUpdates` with a `CloudflareBypassException`; this one is about entries that never enter the
queue.

## Fix

`LibraryUpdateWorker.kt:453`:

```kotlin
internal fun shouldApplyAutoRestrictions(isManualRefresh: Boolean): Boolean = !isManualRefresh
```

`doWork()` now passes the origin (`LibraryUpdateWorker.kt:134`):

```kotlin
addMangaToQueue(categoryId, isManualRefresh = tags.contains(WORK_NAME_MANUAL))
```

and `addMangaToQueue(categoryId: Long, isManualRefresh: Boolean = false)` gates at
`LibraryUpdateWorker.kt:190`:

```kotlin
val restrictions = if (shouldApplyAutoRestrictions(isManualRefresh)) {
    libraryPreferences.autoUpdateMangaRestrictions.get()
} else {
    emptySet()
}
```

## What deliberately did not change

- `ONLY_FETCH_ONCE` and the `!it.manga.initialized` bootstrap stay outside the gate. They are
  data-correctness rules (they avoid re-fetching a chapter list that has never been fetched), not
  cost controls, so a manual refresh still honours them.
- Category inclusion/exclusion (`updateCategories` / `updateCategoriesExclude`) still applies to
  both origins — that is an explicit user selection, not a heuristic.
- The scheduled worker is unchanged: it still applies every filter, because all four restrictions
  exist to keep the periodic background refresh cheap.
- Per-entry update strategies (`ALWAYS_UPDATE`, `ONLY_FETCH_ONCE`, `ONLY_FETCH_NEW_CHAPTERS`) are
  unaffected. They were never part of `autoUpdateMangaRestrictions`.

## Tests

`app/src/test/java/eu/kanade/tachiyomi/data/library/LibraryUpdateRestrictionTest.kt` (JUnit 5, 4
tests) pins the decision: a scheduled refresh applies the filters, a manual refresh ignores them, the
outcome depends only on the origin, and the per-entry data-correctness rules survive a manual
refresh. All pass.
