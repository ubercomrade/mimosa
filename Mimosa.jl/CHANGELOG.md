# Changelog

All notable changes to Mimosa.jl are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — Extensibility API (ADR 0003, Extensibility API Plan)

- New public model geometry contract: `modelname`, `motif_length`,
  `left_context`, `right_context` are the required primary accessors.
  `window_size`, `npositions`, and `site_start_offset` are now derived
  from these and have generic fallback implementations.
- New minimal scan extension point
  `scan_pair_kernel!(forward, reverse, model, sequence, n_positions)`.
  A custom model that subtypes `AbstractMotifModel` and implements
  `modelname`, `motif_length`, and `scan_pair_kernel!` participates in
  `scan`, `prepare_profile`, `compare`, `selectsites`, and
  `reconstruct_pfm` through the public API only.
- New `validate_model(model; capability=:compare)` runtime interface
  check. Supported capabilities: `:compare`, `:sites`, `:cache`.
  Returns the model on success and
  throws `ModelInterfaceError` on failure.
- New `ModelInterfaceError <: MimosaError` carrying the capability,
  model type, and a human-readable diagnostic.
- New exports: `modelname`, `left_context`, `right_context`,
  `scan_pair_kernel!`, `validate_model`, `ModelInterfaceError`.
- New ADR 0003 documenting the model geometry contract.
- New characterization tests pinning the built-in geometry identities
  for all five built-in model families (`test/unit/test_model_geometry.jl`).
- New extension tests covering minimal/context models, validation,
  scan, batch, threaded equivalence, prepare/compare, custom/built-in
  comparison, sites, PFM reconstruction, fingerprint capability
  diagnostics, and worker exception propagation
  (`test/unit/test_extending.jl`).
- New downstream contract test defining a custom model in a separate
  module that imports Mimosa as a regular dependency
  (`test/downstream/runtests.jl`).

### Changed

- Generic workflows (`scan`, `prepare_profile`, `compare`, `build_null`,
  `selectsites`, `reconstruct_pfm`, CLI `inspect-model`/`convert-model`)
  now call the public accessors (`modelname`, `motif_length`,
  `left_context`, `right_context`, `site_start_offset`) instead of
  accessing `model.name`/`model.motif_length`/`model.order`/`model.span`
  on `AbstractMotifModel` values. Built-in concrete types keep direct
  field access inside their own constructors, kernels, and codecs.
- `selectsites` and `reconstruct_pfm` now have a single generic
  `AbstractMotifModel` method. The previous `AbstractHigherOrderMotif`
  dispatch is replaced by the generic method using the public
  geometry; `PWM` keeps its more-specific dispatch.
- `content_fingerprint(model::AbstractMotifModel)` now dispatches on
  concrete built-in types via `_write_model_fingerprint_body!`. Built-in
  byte representations and cache keys are bit-stable. Custom model
  types that need cache/null capability implement `model_fingerprint`
  explicitly; the generic fallback raises a clear `ModelInterfaceError`
  via `validate_model(:cache)` instead of an opaque `ArgumentError`
  from deep inside `content_fingerprint`.
- `is_scannable(::AbstractMotifModel)` now defaults to `true`. The
  function remains a transitional compatibility shim; subtyping
  `AbstractMotifModel` is itself the scannability declaration.
- Removed the intermediate `AbstractMatrixMotif` and
  `AbstractHigherOrderMotif` types. Built-in optimized scan dispatch is
  closed over the five concrete model families; custom models subtype
  `AbstractMotifModel` directly.

### Documentation

- `docs/src/extending.md` rewritten around the minimal contract, with
  examples for both custom models and external score adapters.
- `docs/src/data_layout.md` documents the public geometry contract and
  the built-in mapping table.
- `docs/src/api.md` adds a "Model geometry and extension contract"
  section and `ModelInterfaceError` to the errors section.
- `docs/src/models.md` and `docs/src/numerical_compatibility.md` updated
  to reference ADR 0003 and the public geometry contract.

### Compatibility

- `window_size`, `npositions`, and `site_start_offset` remain public
  but are now derived by default. Existing built-in overrides remain
  and satisfy the ADR 0003 identities.
- `order` and `span` remain fields of their concrete built-in structs
  and of portable bundle manifests; they are not part of the
  `AbstractMotifModel` contract.
- `ScoreProfile(name, scores)` continues to work.
- Symbolic `readmodel(...; format=:...)` remains a boundary adapter.
- Existing model/null/cache format versions are unchanged: model 2,
  null 3, cache 2, annotated-result schema 1.
- Julia 1.10 remains the minimum supported version.
