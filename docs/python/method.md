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
stored value uses the empirical upper-tail count:

```text
-log10(count(calibration_score >= score) / n_calibration)
```

When the score exceeds every calibration score, the table clamps it to the
maximum calibration score rather than returning an infinite value. An empty
calibration table produces zero.

The denominator is the full calibration count and normalized profile values are
Float32. `EmpiricalLogTail` is the sole normalization strategy; `min_logerr`
only selects contributing positions and does not change normalized values.
Legacy hybrid normalization and its `bins` parameter are no longer supported.
Prepared-profile cache entries from older formats are misses, and null
distributions with older normalization identifiers must be rebuilt.
Because both strands share one
calibration, approximately a fraction `10^-t` of positions reaches a value
`t`; tied scores and finite precision can change the exact count.

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
itself scores `1` at offset `0` if its eligible union is nonempty, and swapping query and target mirrors the
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

## Pair-specific Monte Carlo

For each ordered pair, the query is fixed and the target is independently
relabeled in each of K replicates. Each replicate draws an alphabet permutation
at each motif position. The `full` group permits all permutations; `gc`
permutes A/T and C/G separately.
For a function f of an oriented window it evaluates f(pi(x)); declared context
outside the motif and N stay fixed. A reverse window is reverse-complemented
before recoding. All occurrences of a coordinate in higher-order terms are
recoded coherently. Built-in PWM, BaMM, Dimont and Slim tables compile
this same operation directly; a generic window function uses input recoding.

Each randomized target is rescanned and renormalized on both strands, and the
same threshold, shifts, orientations and maximum Jaccard calculation are
applied as for the observed pair.

On uniform iid DNA, coordinate-wise bijections preserve the complete scalar
score distribution of individual windows for arbitrary deterministic f. They
do not preserve literal GC preferences, reverse-complement palindromicity or
all dependence between overlapping windows. A biological null hypothesis and
tail/FDR calibration do not follow from this invariance. Dependence across
models' responses, motif properties, and maximization over alignments can
change the Jaccard reference distribution even after ERR normalization.

The Monte Carlo upper-tail p-value is:

```text
(1 + count(random_score >= observed_score)) / (K + 1)
```

The CLI adds p-values, BH-adjusted p-values, and E-values. By
default, the E-value multiplier is the number of results; pass
`effective_number_of_targets=` to override it. BH adjustment is unaffected by
that override.
