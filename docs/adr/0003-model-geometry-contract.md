# ADR 0003: Model Geometry Contract for Extensibility

Status: Accepted (2026-07-14)

## Context

Mimosa.jl historically exposed model geometry through a mix of public
accessors (`motif_length`, `window_size`, `site_start_offset`), structural
field access (`model.motif_length`, `model.order`, `model.span`), and
internal traits (`context_length`, `scan_width`, `kmer`).  Downstream
model extensions had to mirror one of the built-in struct shapes and
override a mixture of public and private functions, which made the
extension contract implicit and brittle.

The Extensibility API Plan (`Mimosa.jl/EXTENSIBILITY_API_PLAN.md`)
requests a small, explicit, public contract that a third-party model
can satisfy to participate in scan, compare, sites, and reconstruction
workflows without modifying Mimosa.jl.

## Decision

Adopt the geometry contract defined by three public functions:

```
motif_length(model)       # length of the returned site (positive Int)
left_context(model) = 0   # bases left of the site needed for one score
right_context(model) = 0  # bases right of the site needed for one score
```

Mimosa.jl computes the derived quantities:

```
window_size(model) =
    left_context(model) + motif_length(model) + right_context(model)

npositions(model, sequence_length) =
    max(sequence_length - window_size(model) + 1, 0)

site_start_offset(model) = left_context(model)
```

### Coordinate meaning

- A scan position denotes the start of the full window (one-based,
  inclusive, in Julia coordinates).
- The motif site begins at `scan_position + left_context(model)`.
- `left_context` and `right_context` are measured in increasing
  sequence coordinates, not relative to motif orientation.
- Forward and reverse scores at the same scan index refer to the same
  physical full window and to the same physical site interval.  The
  reverse kernel is responsible for orienting the score computation.
- Reverse-complement site extraction reverses only the orientation of
  the returned bases; the underlying physical interval is the same.

If a future model needs different physical site intervals per
orientation, that is outside this contract and requires a separate ADR.

### Built-in mapping

| Model  | `motif_length`        | `left_context` | `right_context` |
|-------|----------------------|----------------|-----------------|
| PWM    | `length(model)`      | 0              | 0               |
| SiteGA | `model.motif_length` | 0              | 0               |
| BaMM   | `model.motif_length` | `model.order`  | 0               |
| Dimont | `model.motif_length` | `model.span`  | 0               |
| Slim   | `model.motif_length` | `model.span`  | 0               |

`order` and `span` remain concrete-type fields and are not part of the
`AbstractMotifModel` contract.  Internal `context_length` continues to
delegate to `left_context` during migration but is no longer an
extension point.

## Consequences

- The minimal model contract for `:compare` is three methods:
  `modelname`, `motif_length`, and `scan_pair_kernel!`.  Context
  methods are needed only when the model actually uses context.
- `window_size`, `npositions`, and `site_start_offset` become derived
  functions with concrete overrides allowed only as a temporary
  migration for built-in types whose geometry is not yet expressible
  via the formula.
- Built-in Float32 scan values, tie-breaking, and coordinate
  conventions are unchanged.  Existing built-in scan, site, and
  comparison results must remain bit-identical.
- Generic workflows must not access built-in-only fields (`name`,
  `representation`, `weights`, `order`, `span`) on `AbstractMotifModel`
  values.  Such access is reserved for the concrete type's own
  constructors, kernels, and codecs.