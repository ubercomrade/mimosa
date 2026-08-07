# Models and Formats

Mimosa has five built-in motif model families and one external score-profile
adapter.

| Model | Public fields | Weight layout | Context |
|---|---|---|---|
| `PWM` | `name`, `weights`, `background` | `(5, motif_length)` | none |
| `BaMM` | `name`, `weights`, `order`, `motif_length` | `(5 ** (order + 1), motif_length)` | `order` on both sides |
| `Dimont` | `name`, `weights`, `order`, `motif_length` | `(5 ** (order + 1), motif_length)` | `order` on both sides |
| `Slim` | `name`, `weights`, `order`, `motif_length` | `(5 ** (order + 1), motif_length)` | `order` on both sides |
| `SiteGA` | `name`, `weights`, `motif_length` | `(25, motif_length)` | none |

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

For an in-memory frequency matrix, use `pwm_from_pfm` and then construct a
`PWM` through the public model API.

## Higher-order models

BaMM `.ihbcp` files are materialized into a 5-ary context matrix. `order` can
be supplied to `read_model` or the lower-level BaMM reader to materialize a
lower order when the source contains higher orders.

SiteGA `.mat` files use 25 dinucleotide rows. Dimont and Slim Jstacs XML files
are parsed and materialized into the same 5-ary row layout used by context
scanning.

## Score profiles

`ScoreProfile` stores precomputed Float32 score rows. It can be prepared and
compared, but it cannot scan raw DNA or provide motif sites.

```python
from mimosa import read_scores

profile = read_scores("examples/scores_1.fasta")
```

Each score row is introduced by a `>` header. Values may be whitespace- or
comma-separated. Ragged row lengths are supported.

## Model bundles

`read_model` recognizes a directory containing `manifest.toml` as a portable
model bundle. Bundles support all five built-in model families. See
[Storage](storage.md) for the on-disk layout and checksums.
