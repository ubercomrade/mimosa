# ADR 0009: Encoded sequence and model constructor validation

## Date

2026-07-12

## Status

Accepted

## Context

PLAN_2.md items B2 and B3 identified that:

1. `EncodedSequenceBatch` did not validate that all bytes in its `data` field
   are in the valid range `0..N_CODE` (0x00..0x04). The scanning kernels
   (`scan_forward!`, `scan_reverse!`, `_ho_scan_forward!`, etc.) use
   `@inbounds` with `Int(seq[i]) + 1` as a row index into representation
   matrices. An invalid code (e.g., 0x05 or 0xFF) would cause an out-of-bounds
   memory access, producing undefined behavior rather than a controlled error.

2. `reverse_complement!(dest, src)` did not check whether `dest` and `src`
   alias the same buffer. In-place reverse complement on an aliased buffer
   would corrupt data mid-copy.

3. Scan kernels did not validate that `dest` has enough elements for `n_pos`
   positions, relying on callers to pre-allocate correctly.

4. PFM had no constructor validation at all. PWM validated weights but not
   background. BaMM/SiteGA/Dimont/Slim validated dimensions and finite values
   but did not guard against excessively high order/span values that would
   cause `5^(order+1)` to allocate gigabytes.

## Decision

### Encoded sequence validation (B2)

- All **public** `EncodedSequenceBatch` constructors validate that every byte
  in `data` satisfies `0 <= byte <= N_CODE`, raising `InvariantError` on
  violation.
- An **internal** unsafe constructor `_unsafe_encoded_batch` uses a
  `Val{:unsafe}` token to skip code validation. It is used only by
  `make_random_sequences`, where codes are drawn from `_BASE_LOOKUP` (0..3)
  and are valid by construction.
- `from_padded` validates the padding value, per-row lengths (non-negative,
  within matrix width), and delegates to the public constructor for code
  validation.
- `reverse_complement!` checks `pointer(dest) == pointer(src)` with matching
  lengths and raises `ArgumentError` on aliasing.
- All scan kernels (`scan_forward!`, `scan_reverse!`, `scan_best!`,
  `scan_both!`, `_ho_scan_forward!`, `_ho_scan_reverse!`, `_ho_scan_best!`,
  `_ho_scan_both!`) validate `n_pos >= 0` and `length(dest) >= n_pos` before
  entering the `@inbounds` loop.
- `extract_site_matrix` validates that `start + motif_width - 1 <=
  length(seq)` before the `@inbounds` extraction loop.
- Invariant comments are placed directly above each approved `@inbounds`
  kernel, documenting why the indexing is safe.

### Model constructor hardening (B3)

- **PFM**: validates 4 rows, positive width, finite values, non-negative
  values.
- **PWM**: validates 5 rows, positive width, finite weights, finite and
  non-negative background, background sum approximately 1.0 (rtol=1e-4).
- **BaMM/SiteGA/Dimont/Slim**: validate order/span >= 0 and <= 10 (guards
  against `5^(order+1)` allocation blow-up), correct row count, positive
  motif_length, finite values. The order/span check is performed **before**
  the `5^(order+1)` exponentiation.

## Consequences

- Public construction of `EncodedSequenceBatch` with invalid codes now raises
  `InvariantError` instead of potentially causing undefined behavior in
  scanning kernels.
- The `Val{:unsafe}` inner constructor is not exported and is only used
  internally; downstream users cannot bypass validation.
- Scan kernel entry points now have O(1) validation overhead (two integer
  comparisons) before the hot loop, which is negligible.
- PFM/PWM/BaMM/SiteGA/Dimont/Slim constructors now reject invalid invariants
  at construction time rather than allowing downstream code to encounter
  `BoundsError`, `NaN`, or allocation failures.
- The order/span upper limit of 10 is a safety guard, not a scientific
  restriction; real motif models use order/span 0-5. A higher limit would
  require an ADR amendment.