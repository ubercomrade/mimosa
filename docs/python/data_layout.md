# Data Layout and Coordinates

## Encoded sequences

`EncodedSequences` stores a batch in one flat `uint8` buffer and one `int64`
offset array:

```text
sequence(i) = data[offsets[i] : offsets[i + 1]]
```

Python indexing is zero-based and the end offset is exclusive. Empty rows are
valid. DNA codes are:

| Base | Code |
|---|---:|
| A | 0 |
| C | 1 |
| G | 2 |
| T | 3 |
| N or ambiguous base | 4 |

Lowercase bases are accepted. Other IUPAC ambiguity codes map to `N`.

## Ragged score arrays

`RaggedArray` uses the same buffer-plus-offset representation for Float32 score
tracks. Its row slices are views into the flat data buffer. `StrandPair` holds
forward and reverse ragged arrays; symmetric profiles may reference the same
object for both strands.

## Model geometry

Every model has a motif length and non-negative left/right context:

```text
window_size = left_context + motif_length + right_context
n_positions(sequence_length) = max(sequence_length - window_size + 1, 0)
site_start = scan_index + left_context
site_end = site_start + motif_length
```

The scan index is the start of the complete scoring window. Forward and reverse
scores at one scan index refer to the same physical interval. Only score
orientation and returned reverse-strand bases differ.

For built-in models:

| Model | Left context | Right context |
|---|---:|---:|
| PWM | 0 | 0 |
| SiteGA | 0 | 0 |
| BaMM | `order` | `order` |
| Dimont | `order` | `order` |
| Slim | `order` | `order` |

## Public coordinates

All public Python and CLI coordinates are zero-based half-open. Scan and anchor
positions are zero-based indices. Site extraction returns the physical motif
interval, excluding model context.

`SiteCollection.strands` uses `0` for forward and `1` for reverse. Reverse
sites are reverse-complemented before PFM reconstruction.

## Numeric types

- Encoded bases use `uint8`.
- Offsets and indices use `int64` in memory.
- Scan and normalized profile values use `float32`.
- Alignment metrics accumulate in `float64` before the result score is stored.
- Model bundles store raw little-endian Float32 arrays.
