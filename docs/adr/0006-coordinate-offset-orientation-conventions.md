# ADR 0006: Coordinates, Offsets, and Orientations

## Status

Accepted and implemented for scanning, sites, and profile-only comparison.

## Coordinates

Julia library coordinates are one-based and inclusive. `SiteHit` ranges remain
inside the source sequence. CLI JSON converts sites to zero-based half-open
`start`/`end` coordinates only at serialization.

Reverse-strand hits retain the forward-sequence coordinates of the physical
window. Their extracted sequence is reverse complemented.

## Scan Orientation

At scan position `p`, forward and reverse scores refer to the same physical
window. Reverse scoring reads that window in reverse-complement orientation;
the score track itself is not reversed. `BestStrand` takes the per-position
maximum.

## Profile Offset

Offset is query displacement relative to target. Positive means the query is
shifted right. Candidate shifts are evaluated from `-search_range` through
`+search_range`.

Within one orientation, ties are resolved by:

1. higher score;
2. more contributing site windows;
3. smaller absolute shift;
4. first visited shift on a complete tie.

Across orientations the same score/site/absolute-shift ordering applies, then
the fixed orientation priority is used:

| Label | Query strand | Target strand | Priority |
|---|---|---|---|
| `++` | forward | forward | 1 |
| `+-` | forward | reverse | 2 |
| `-+` | reverse | forward | 3 |
| `--` | reverse | reverse | 4 |

Precomputed `ScoreProfile` inputs may collapse redundant strand candidates when
forward and reverse storage is identical, without changing the winning result.

## Site Ordering

Site collections are deterministic: sequence index ascending, score descending,
start ascending, then forward before reverse. Thread count must not change this
order or any coordinate/orientation field.
