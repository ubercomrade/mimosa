# Python Format Inventory

> **Stage 0 audit artifact.** Detailed description of every file format read or written by the
> Python implementation. Julia parsers must accept the same valid inputs and produce equivalent
> in-memory representations. Malformed inputs must produce typed errors.

## 1. MEME format

**Extension:** `.meme`
**Reader:** `io/meme.py` — `read_meme(path, index=0)`, `read_meme_many(path)`
**Writer:** none

### Structure

```
MEME version 4

ALPHABET= ACGT

Background letter frequencies
A 0.31 C 0.19 G 0.19 T 0.31

MOTIF pwm_model
letter-probability matrix: alength= 4 w= 12 nsites= 595
0.00003100	0.92578200	0.06339200	0.01079600
0.999931000	0.000019000	0.000019000	0.000031000
...
```

### Parsing logic

1. Scan line-by-line for `MOTIF` lines.
2. Extract motif name from second whitespace-delimited token.
3. Read next line, parse `w=<length>` from the header.
4. Read `<length>` rows of 4 float values each.
5. Validate: shape must be `(length, 4)`, all finite.
6. Transpose to `(4, length)` — i.e., `[base, position]` layout.
7. Cast to `float32`.

### Invariants to preserve

- Output array shape: `(4, width)` where width = motif length.
- Orientation: `[base, position]` (rows = A/C/G/T, cols = positions).
- Multi-motif files preserve file order.
- `read_meme_many` raises if no motifs found.
- `read_meme(path, index)` raises `IndexError` if index out of range.

### Malformed cases to handle

- Missing `MOTIF` name (line has only "MOTIF").
- `w=` missing or zero/negative.
- Fewer rows than declared `w`.
- Non-finite values in matrix.
- Empty file (no MOTIF lines).

### Julia requirements

- Size limits on declared width (reject implausibly large motifs).
- Validate all values are finite.
- Preserve `[base, position]` output orientation.
- `Float32` after parsing (parser may accept `Float64` input).

## 2. PFM format

**Extension:** `.pfm`
**Reader:** `io/pfm.py` — `read_pfm(path)`
**Writer:** `io/pfm.py` — `write_pfm(pfm, name, length, path)`

### Structure

```
>pwm_model
0.000031000	0.925782000	0.063392000	0.010796000
0.999931000	0.000019000	0.000019000	0.000031000
...
```

Lines starting with `>` are treated as comments and skipped by `np.loadtxt`. The matrix is parsed as whitespace-delimited floats.

### Parsing logic

1. `np.loadtxt(path, comments=">", dtype=np.float32)`.
2. If 1D, reshape to `(1, -1)`.
3. Must be 2D.
4. If `shape[1]` is 4 or 5 (nucleotide rows), transpose → `(4 or 5, width)`.
5. If `shape[0]` is 4 or 5, keep as-is.
6. Otherwise, raise "one axis must contain 4 or 5 nucleotide rows."
7. Validate: positive width, all finite.
8. Cast to `float32`.

### Writer

```
>{name}
{pfm.T formatted as tab-delimited, 6 decimal places}
```

The writer transposes back to `(position, base)` layout for human-readable output.

### Invariants to preserve

- Auto-detect orientation: 4 or 5 rows = `[base, position]`; 4 or 5 columns = `[position, base]`.
- Name extracted from file path stem (not from `>` header line — Python uses `os.path.splitext(os.path.basename(path))[0]`).
- 5-row PFM includes ambiguous state; Julia may trim to 4 rows or preserve depending on ADR.

## 3. BaMM format (`.ihbcp`)

**Extension:** `.ihbcp` (also `.ihbp` companion file, ignored)
**Reader:** `io/bamm.py` — `parse_file_content(path)`, `read_bamm(path, target_order)`
**Writer:** `joblib.dump` (not a format writer — pickle)

### Structure

Position blocks separated by blank lines. Each block contains `max_order + 1` lines, one per order `k=0..max_order`. Line for order `k` has `4^(k+1)` whitespace-delimited float values.

```
# comment lines start with #
0.25 0.25 0.25 0.25          # order 0: 4 values
0.0625 0.0625 ... 0.0625     # order 1: 16 values
...
```

### Parsing logic

1. Read entire file, split into blocks by `\n\n`.
2. Strip comments (`#`) and empty lines.
3. Each block → list of arrays (one per order).
4. Validate: all positions have the same number of order lines.
5. `max_order = len(block[0]) - 1`.
6. Group by order: `data_by_order[k] = [block[pos][k] for pos in range(num_positions)]`.
7. Validate: each order `k` array has `4^(k+1)` values.

### Conversion to log-odds tensor

1. For each position, compute `current_k = min(pos, target_order)`.
2. Extract probability array `p_motif` for order `current_k`.
3. Compute `log_odds = log((p_motif + 1e-10) / (uniform_bg + 1e-10))` where `uniform_bg = 0.25^(current_k+1)`.
4. Reshape to `[4] * (current_k + 1)` tensor.
5. If `current_k < target_order`, broadcast to `[4] * (target_order + 1)`.
6. Stack positions along last axis → `[4, 4, ..., 4, length]`.
7. Compute per-position min scores: `min_scores = min over all alphabet axes`.
8. Build 5-ary tensor: `[5, 5, ..., 5, length]` where index 4 (N) on each axis = min over indices 0-3.

### Invariants to preserve

- Output tensor shape: `[5] * (target_order + 1) + [length]`, dtype `float32`.
- N axis values = minimum over concrete nucleotides on that axis.
- `.ihbp` companion file is ignored (contains background frequencies).
- `target_order` truncation: if requested order exceeds file max, use max.
- Uniform background: `0.25^(k+1)` — not read from file.

### Security concerns

- No size limit on declared `max_order` (a malicious file could declare huge orders → `5^(order+1)` tensor).
- No limit on number of positions.
- Julia must enforce size limits and reject implausible dimensions.

## 4. SiteGA format (`.mat`)

**Extension:** `.mat`
**Reader:** `io/sitega.py` — `read_sitega(path)`
**Writer:** `io/sitega.py` — `write_sitega(model, path)`

### Structure

```
Bootatrap
42	LPD count
12	Model length
0.000000000000	Minimum
3.765757980263	Razmah
0	2	0.042273837993	0	aa
3	3	0.070166413447	0	aa
...
```

Line 1: model name. Line 2: LPD count (ignored). Line 3: model length. Lines 4-5: minimum/maximum scores (stored in config). Subsequent lines: `start stop value dinuc_index dinucleotide` where dinucleotide is two lowercase letters from `acgt`.

### Parsing logic

1. Read name, skip LPD count, read length.
2. Initialize `5×5×length` float32 zero matrix.
3. For each segment line: parse `start, stop, value, _, dinucleotide`.
4. Validate: `0 <= start <= stop < length`, valid dinucleotide.
5. Distribute value: `sitega[nuc1][nuc2][index] += value / (stop - start + 1)` for each index in `[start, stop]`.

### Writer logic

1. Compute min/max from representation.
2. For each non-zero `(nuc1, nuc2)` pair, find maximal constant-value segments.
3. Write segments with `total_value = value * range_length`.
4. Dinucleotide index: `itertools.product("acgt", repeat=2)` order (00=aa, 01=ac, 02=ag, 03=at, 10=ca, ...).

### Invariants to preserve

- Tensor shape: `(5, 5, length)`, dtype `float32`.
- 5-ary: N state (index 4) is zero (not filled with min — unlike BaMM/Dimont/Slim).
- Value distribution: segment value divided by range length, accumulated per position.
- Score bounds stored in config: `minimum`, `maximum`.

## 5. Dimont XML format

**Extension:** `.xml`
**Reader:** `io/xml.py` — `read_dimont(path)`
**Writer:** `joblib.dump` only

### Structure

Jstacs XML with `ThresholdedStrandChIPper/function/pos/MarkovModelDiffSM` root. Contains `bayesianNetworkSF/trees` with parameter trees per position. Each tree has context positions, a root `treeElement`, and recursive `children` with `treeElement` nodes. Leaf nodes have `pars/pos/parameter` entries with `symbol` and `value`.

### Parsing logic

1. Parse XML tree.
2. Extract `contextPoss` per position to determine span.
3. Recursively walk parameter tree: at each node, either `scores` (leaf) or `children` (4 subtrees indexed by `val`).
4. For each concrete context (4^span combinations), walk tree to find leaf scores.
5. Build dense tensor: `[5] * (span + 1) + [length]`.
6. N axis filled with `min` over concrete nucleotides.

### Invariants to preserve

- Output tensor shape: `[5] * (span + 1) + [length]`, dtype `float32`.
- Log-odds: `scores + log(4)` added to leaf values (uniform background).
- N on context axes: `min` over A/C/G/T on that axis.
- N on output axis: `min` over A/C/G/T on the score axis.

### Security concerns

- XML parsed without entity restrictions → XXE attacks possible.
- No size limits on declared length or span.
- Julia must use a safe XML parser with entity restrictions and size limits.

## 6. Slim XML format

**Extension:** `.xml`
**Reader:** `io/xml.py` — `read_slim(path)`
**Writer:** `joblib.dump` only

### Structure

Jstacs XML with `SLIM` root element containing `length`, `distance`, `componentMixtureParameters`, `ancestorMixtureParameters`, `dependencyParameters` arrays.

### Parsing logic

1. Parse `length`, `componentMixtureParameters`, `ancestorMixtureParameters`, `dependencyParameters`.
2. Compute `span` from component/ancestor structure: `max(component_index + ancestor_count - 1)`.
3. Normalize parameters to log-probabilities: `log_normalize(x) = x - logsumexp(x)`.
4. For each position, for each concrete context (4^span), compute symbol log-probabilities:
   - Component 0: `component_log_prob + dependency[0, symbol]`.
   - Components 1+: logsumexp over ancestors of `component_log_prob + dependency[context, symbol]`.
   - Total: `logsumexp(component_scores) + log(4)`.
5. Build dense tensor: `[5] * (span + 1) + [length]`.
6. N axis handling: same as Dimont.

### Invariants to preserve

- Output tensor shape: `[5] * (span + 1) + [length]`, dtype `float32`.
- Log-sum-exp computation must be numerically stable.
- Span computed from model structure, not stored explicitly.
- `alphabet_size` from `len(dependency_params[0][0][0])` — expected to be 4.

## 7. Score FASTA format

**Extension:** `.fasta` (or any extension)
**Reader:** `io/batches.py` — `read_scores(path)`
**Writer:** none

### Structure

```
>SEQ_00001 generated_by_numpy
0.259 0.72  0.34  0.674 ...
0.821 0.557 0.847 0.941 ...
>SEQ_00002 generated_by_numpy
0.817 0.332 ...
```

FASTA-like format where each "sequence" is a series of whitespace-delimited float values. Values may span multiple lines. Commas are replaced with spaces before splitting.

### Parsing logic

1. Read line by line.
2. `>` starts a new profile; accumulate values for the current profile.
3. Non-header lines: `float(token)` for each whitespace-delimited token.
4. Commas replaced with spaces (allows comma/space mixed delimiters).
5. Invalid float → `ValueError` with file path and line.
6. Profiles packed into `MaskedBatch` (float32, padding 0.0).

### Invariants to preserve

- Profiles may have different lengths → padded dense batch with mask.
- dtype: `float32`, padding value: `0.0`.
- Description after `>` is ignored (only the profile values matter).
- Empty profiles (no values before next `>`) are skipped.

## 8. DNA FASTA format

**Extension:** `.fa`, `.fasta`, or any
**Reader:** `io/batches.py` — `read_fasta(path)`
**Writer:** none

### Parsing logic

1. Build translation table: `A=0, C=1, G=2, T=3, a=0, c=1, g=2, t=3`, everything else = 4.
2. Read line by line.
3. `>` starts a new sequence; accumulate bytes.
4. Sequence lines: encode ASCII bytes through translation table.
5. Pack into `SequenceBatch` (int8, padding 4).

### Encoding

| Character | Code |
|---|---|
| A, a | 0 |
| C, c | 1 |
| G, g | 2 |
| T, t | 3 |
| N, n, and all others | 4 |

### Invariants to preserve

- Lowercase normalized to uppercase equivalents.
- IUPAC ambiguity codes (R, Y, S, W, K, M, B, D, H, V) are NOT treated specially — they map to 4 (N).
- Empty sequences are allowed (length 0).
- Sequences shorter than motif width produce 0 scan positions.
- Encoding is int8, padding value is 4.

## 9. Joblib/Pickle (legacy)

**Extension:** `.pkl`, `.joblib`
**Reader:** `handlers._load_pickled_generic_model` (trusted only)
**Writer:** `handlers._dump_model` (joblib.dump)

### Status

This is the current persistence format for BaMM, Dimont, Slim models and null distributions. It is **not** being ported to Julia as a storage format. A separate converter will be provided for migrating existing files.

### Security

- `joblib.load` can execute arbitrary code.
- Only trusted inputs should be loaded.
- Julia converter will require explicit `--trusted-input` flag.

## 10. DIST format

**Extension:** `.dist`
**Reader:** none
**Writer:** `io/dist.py` — `write_dist(threshold_table, max_score, min_score, path)`

Not used in the main comparison pipeline. Included for completeness. Deferred to post-1.0 in Julia.