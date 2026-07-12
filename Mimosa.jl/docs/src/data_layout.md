# Data Layout

## Matrix layout

Mimosa.jl uses **column-major Julia** layout consistently.

### PWM weights

`weights[base, position]` with `base ∈ 1:5` (A, C, G, T, N):

```
         pos1   pos2   pos3   ...
A  (1)  w11    w12    w13    ...
C  (2)  w21    w22    w23    ...
G  (3)  w31    w32    w33    ...
T  (4)  w41    w42    w43    ...
N  (5)  w51    w52    w53    ...
```

The N row holds the per-column minimum over concrete bases, matching the
Python representation.

### PFM frequencies

`frequencies[base, position]` with `base ∈ 1:4` (A, C, G, T).

### Higher-order representation

`representation[context_code, position]` where `context_code` is a 5-ary
encoded index. For BaMM with order `k`, the shape is `(5^(k+1), motif_length)`.
For Dimont/Slim with span `s`, the shape is `(5^(s+1), motif_length)`.

### SiteGA representation

`representation[dinuc_code, position]` where `dinuc_code = base1 * 5 + base2`
(0-indexed), giving shape `(25, motif_length)`.

## Sequence encoding

5-ary encoding with flat buffer:

| Base | Code |
|------|------|
| A | 0x00 |
| C | 0x01 |
| G | 0x02 |
| T | 0x03 |
| N / ambiguous / padding | 0x04 |

Lowercase is normalized to uppercase. All IUPAC ambiguous codes map to 0x04.
Sequences are stored in `EncodedSequenceBatch` as a flat `UInt8` buffer with
offsets — no padding, no rectangular matrix.

## Coordinate conventions

### Internal Julia coordinates

- **One-based inclusive** indexing throughout the library
- Site coordinates use `UnitRange{Int}`: `start:stop` (both inclusive)

### CLI JSON coordinates

- **Zero-based half-open** `start`/`end` coordinates in JSON output
- Conversion happens only at the serialization boundary
- This matches the Python CLI contract

## Offset convention

Offset is the displacement of the query relative to the target:
- **Positive** = query shifted right
- **Negative** = query shifted left
- Iteration goes from **negative to positive** offsets
- **First offset wins** on equal score (deterministic tie-breaking)

## Orientation convention

Four orientation candidates with tie-break ranks:

| Orientation | Rank | Meaning |
|------------|------|---------|
| `++` | 0 | Query forward, target forward |
| `+-` | 1 | Query forward, target reverse |
| `-+` | 2 | Query reverse, target forward |
| `--` | 3 | Query reverse, target reverse |

Lower rank wins on equal score.

## Reverse complement

For a PWM `weights[base, position]`:
1. Flip base rows: A↔T (rows 1↔4), C↔G (rows 2↔3), N stays (row 5)
2. Reverse position columns

For an encoded sequence `seq[i]`:
- Complement: `comp(b) = 3 - b` for b ∈ {0,1,2,3}; N (4) stays 4
- Reverse: `rc[i] = comp(seq[n - i + 1])`