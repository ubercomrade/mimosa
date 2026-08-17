# Method and Statistics

Mimosa compares models by their behavior on a shared sequence batch.

## Comparison pipeline

1. Scan the sequences with both models to obtain forward and reverse score tracks.
2. Normalize scores to empirical `-log10(ERR)` values. The comparison batch is the calibration batch unless `background=` is supplied.
3. Select one strict best anchor per non-empty row when `min_logerr <= 0`, or every score above the threshold when it is positive.
4. Align site-centered profile windows over the configured shifts and strand orientations.
5. Select the best result and report its score, offset, orientation, and contributing site count.

The four orientation labels are `++`, `+-`, `-+`, and `--`. Offset is the target
physical site displacement relative to the query; positive values move the
target right. Model scoring context does not affect this public coordinate.
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
only selects anchors.

## Similarity metrics

For each contributing aligned row, continuous overlap is:

```text
CO(x, y) = sum(min(x, y)) / min(sum(x), sum(y))
```

Dice similarity is calculated for each contributing row as:

```text
Dice(x, y) = 2 * sum(min(x, y)) / (sum(x) + sum(y))
```

`co` and `dice` average the finite per-row values. `cosine` calculates a
per-row cosine similarity and averages it.

## Sites and PFMs

Sites are selected from raw forward/reverse scan tracks. A reverse-strand site
is reverse-complemented before PFM counting. Ambiguous bases are skipped. PFM
reconstruction defaults to a pseudocount of `0.25`.

## Null distributions

`build_null` is currently PWM-only. It creates one shuffled PWM per source
model, prepares original and shuffled profiles once, and samples eligible
ordered pairs. A fresh NumPy `default_rng` stream controls model shuffling and
pair sampling. The sampling contract is identified in the stored bundle.

The empirical upper-tail p-value is:

```text
(1 + count(null_score >= observed_score)) / (n_null + 1)
```

`annotate_results` adds p-values, BH-adjusted p-values, and E-values. By
default, the E-value multiplier is the number of results; pass
`effective_number_of_targets=` to override it. BH adjustment is unaffected by
that override.
