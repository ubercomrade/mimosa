# Models and Formats

Mimosa has four available built-in motif model families and one external score-profile
adapter.

| Model | Public fields | Weight layout | Context |
|---|---|---|---|
| `PWM` | `name`, `weights`, `background` | `(5, motif_length)` | none |
| `BaMM` | `name`, `weights`, `order`, `motif_length` | `(5 ** (order + 1), motif_length)` | `order` on both sides |
| `Dimont` | `name`, `weights`, `order`, `motif_length` | `(5 ** (order + 1), motif_length)` | `order` on both sides |
| `Slim` | `name`, `weights`, `order`, `motif_length` | `(5 ** (order + 1), motif_length)` | `order` on both sides |

The `SiteGA` implementation is retained but temporarily disabled, including
construction, reading and scanning.

All built-in model values are frozen dataclasses with contiguous read-only
Float32 arrays. `Dimont.order` and `Slim.order` describe the materialized
context order; they are not exposed as `span`.

## PWM and PFM

MEME and plain PFM files are converted to `PWM`. A PWM has rows for A, C, G,
T, and N. The N row is the per-column minimum of the four concrete nucleotide
rows. Plain PFM input has one position per row and four columns in A/C/G/T
order.

```python
from mimosa import read_model

pwm = read_model("examples/foxa2.meme")
pfm_pwm = read_model("examples/pif4.pfm", format="pfm", background=0.25)
```

For an in-memory frequency matrix, `pwm_from_pfm(pfm, name="my-model")`
returns a `PWM` directly. The matrix has shape `(4, motif_length)`, with
finite probabilities in `[0, 1]` and columns summing to one. Supply a non-empty
name. Conversion uses the natural logarithm `log((pfm + 1e-4) / background)` and a uniform scalar
background of `0.25`. This background parameter is unrelated to the sequence
batch used for score normalization.

## Higher-order models

BaMM `.ihbcp` files are materialized into a 5-ary context matrix. `order` can
be supplied to `read_model` or the lower-level BaMM reader to materialize a
lower order when the source contains higher orders.
The Python text reader caps a requested order at the highest available order;
the CLI instead rejects an unavailable requested order. Model bundles preserve
their stored order and are not rematerialized by `read_model(..., order=...)`.

Dimont and Slim Jstacs XML files are parsed and materialized into the same
5-ary row layout used by context scanning.

## Score profiles

`ScoreProfile` stores precomputed Float32 score rows. It can be prepared and
compared, but it cannot scan raw DNA or provide motif sites.

```python
from mimosa import read_scores

profile = read_scores("examples/scores_1.fasta")
```

Each score row is introduced by a `>` header. Values may be whitespace- or
comma-separated. Ragged row lengths are supported. Rows are raw scores and
are normalized during preparation; they are not assumed to be logERR values
already. The same scores are used for both strands. Row headers mark boundaries;
`ScoreProfile.name` is derived from the file stem, not from the headers.

## Model bundles

`read_model` recognizes a directory containing `manifest.toml` as a portable
model bundle. Reading and writing currently support PWM, BaMM, Dimont and Slim;
legacy SiteGA bundles are rejected while that model is disabled. See
[Storage](storage.md) for the on-disk layout and checksums.
