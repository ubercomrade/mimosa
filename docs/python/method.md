# Method and Statistics

Mimosa compares models by their behavior on a shared sequence batch.

## Comparison pipeline

1. Scan the sequences with both models to obtain forward and reverse score tracks.
2. Normalize scores to empirical `-log10(ERR)` values. The comparison batch is the calibration batch unless `background=` is supplied.
3. Collect every position whose value reaches `min_logerr`.
4. Score the weighted Jaccard similarity for every circular shift and strand orientation.
5. Select the best result and report its score, offset, orientation, and contributing site count.

Comparison treats every sequence as circular. Each sequence of length `L`
yields exactly `L` values per strand, and the value at index `i` belongs to the
site whose first motif base is base `i`; higher-order models read their context
from the ring. All models therefore share one coordinate system and no
positions are lost at sequence ends.

The four orientation labels are `++`, `+-`, `-+`, and `--`. Offset is the
circular shift `s` that pairs query position `i` with target position
`(i + s) mod L`; positive values move the target right. Model scoring context
does not affect this public coordinate.
Final ties are resolved by larger score, larger `n_sites`, smaller absolute
offset, and orientation rank in the order listed above. Within an orientation,
shifts are visited from most negative to most positive.

## Normalization

Prepared profiles use one threshold-independent exact empirical table. Every
stored value therefore uses:

```text
-log10(count(calibration_score >= score) / n_calibration)
```

The denominator is the full calibration count and normalized profile values are
Float32. `HybridEmpiricalLogTail` remains the default strategy identifier and
the low-level `fit(..., tail_logerr=...)` helper still supports hybrid tables,
but `prepare_profile` canonicalizes the complete exact table so `min_logerr`
only selects the contributing positions. Because both strands share one
calibration, a fraction `10^-t` of the positions of any model reaches a value
`t`.

## Weighted Jaccard similarity

The only similarity measure is the weighted Jaccard index, reported as the
metric `jaccard`. For one alignment, let `u_i` be the query value at position
`i` and `v_i` the target value at the paired position, pooled over all rows:

```text
S       = { i : max(u_i, v_i) >= min_logerr }
score   = sum(min(u_i, v_i) for i in S) / sum(max(u_i, v_i) for i in S)
n_sites = |S|
```

The score is `0` when `S` is empty and lies in `[0, 1]`. A model compared with
itself scores `1` at offset `0`, and swapping query and target mirrors the
offset and orientation without changing the score. Both profiles must have the
same row lengths.

`min_logerr` must be positive and defaults to `4.0`; `search_range` defaults to
`5`. About `positions * 10^-min_logerr` positions per strand pass the threshold,
so the defaults are tuned for `10^6` positions per strand (10 000 sequences of
100 bp). Mimosa warns when fewer than 50 are expected: use more sequence or a
lower threshold. The measure was validated on uniform random sequences
calibrated on themselves.

## Sites and PFMs

Sites are selected from raw forward/reverse scan tracks. A reverse-strand site
is reverse-complemented before PFM counting. Ambiguous bases are skipped. PFM
reconstruction defaults to a pseudocount of `0.25`.

## Null distributions

`build_null` is currently PWM-only. It creates one shuffled PWM per source
model, prepares original and shuffled profiles once, and samples eligible
ordered pairs. A fresh NumPy `default_rng` stream controls model shuffling and
pair sampling. Null profiles are circular and scored with the same weighted
Jaccard measure, so a null distribution only annotates comparisons that use the
same sequences, background, `search_range`, and `min_logerr`. The sampling
contract is identified in the stored bundle.

The empirical upper-tail p-value is:

```text
(1 + count(null_score >= observed_score)) / (n_null + 1)
```

`annotate_results` adds p-values, BH-adjusted p-values, and E-values. By
default, the E-value multiplier is the number of results; pass
`effective_number_of_targets=` to override it. BH adjustment is unaffected by
that override.
