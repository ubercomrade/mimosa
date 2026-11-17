# Custom Models

Mimosa exposes small operation-local extension contracts. Custom extensions do
not require editing Mimosa or registering global plugins.

## Custom motif model

A custom model subclasses `MotifModel` and provides `name`, `motif_length`, and
`scan_into`. Context models may also provide `left_context` and
`right_context`.

```python
import numpy as np

from mimosa import MotifModel


class ConsensusModel(MotifModel):
    name = "consensus"
    motif_length = 4

    def __init__(self, pattern):
        self.pattern = np.asarray(pattern, dtype=np.uint8)

    def scan_into(self, sequence, forward, reverse, /):
        reverse_pattern = self.pattern[::-1].copy()
        concrete = reverse_pattern != 4
        reverse_pattern[concrete] = 3 - reverse_pattern[concrete]
        for position in range(forward.size):
            window = sequence[position : position + self.motif_length]
            forward[position] = np.count_nonzero(window == self.pattern)
            reverse[position] = np.count_nonzero(window == reverse_pattern)
```

The caller owns `forward` and `reverse`; the implementation fills both
Float32 arrays for every complete model window. Input sequences are validated
`uint8` arrays with codes `0..4`.

The custom model participates in `scan`, `prepare_profile`, `compare`,
`select_sites`, and `reconstruct_pfm`. Comparison scans circular sequences by
passing `scan_into` each sequence wrapped with `left_context` bases on the left
and `window_size - left_context - 1` bases on the right, so no circular logic
is needed in the model. A stable SHA-256 `fingerprint()` is
needed for the prepared-profile cache and null construction. Arbitrary custom models are
not accepted by `write_model` because `scan_into` does not define how to
serialize model parameters.

## Batched fixed-window functions, including CNNs

`WindowModel` handles scanning geometry and batching for a function that
scores complete oriented input windows:

```python
from mimosa import WindowModel, relabel_model

model = WindowModel(
    "assayed-tf", motif_length=20, score_windows=my_scorer,
    left_context=10, right_context=10,
    fingerprint="stable-hash-of-scoring-code-and-learned-parameters",
    calibration_group="my-cnn-architecture-and-context-v1",
)
random_model = relabel_model(model, seed=112)
```

The public direct Monte Carlo implementation currently supports built-in
models only, not `WindowModel`.

`my_scorer(windows)` receives a two-dimensional uint8 array of shape
`(n, left_context + motif_length + right_context)`. It returns a real NumPy
vector of shape `(n,)`, finite after conversion to Float32. A framework adapter
may convert to one-hot input, move it to a GPU, choose one specified output,
and return CPU scores. Neural models must be in deterministic inference mode;
dropout, training-mode batch normalization and updates between calls violate
the function contract. Outputs must be independent of batch size and other
windows in the batch.

Rows are batched together without mixing their windows. `batch_size` caps
the number of windows and a separate internal element limit bounds temporary
input arrays for wide contexts. Device tensors and framework allocations are
owned by the callback. This is an interface for neural scorers, not a neural
checkpoint reader or a guarantee of an efficient whole-sequence CNN scan.
Recoding windows separately may prevent reuse of overlapping convolution work.

For `relabel_model`, one independent alphabet permutation is fixed at every
motif coordinate and reused for every window. Declared context coordinates
and N remain fixed. Reverse-complementing precedes recoding. On uniform iid
DNA this preserves the scalar score distribution of individual windows;
GC identity, palindrome structure and overlapping-window dependence need
not be preserved. The original learned parameters are unchanged. The
composite function need not be representable by the original architecture
without its input wrapper.

For custom `MotifModel` implementations, input relabeling uses the forward
single-window scorer and canonical reverse-complement semantics. A model with
different strand semantics should expose a suitable `WindowModel` function
explicitly. Without a batched window callback, scanning arbitrary custom
models on each recoded window can be slow. A custom null additionally needs
an explicit `calibration_group` string and stable `fingerprint()`; library
callers check both compared models' group before annotating scores.

Native PWM/BaMM/Dimont/Slim surrogates compile the same input recoding
into tables. Only the generic fallback needs explicit window enumeration.

## ScoreProfile

An external scanner that already produces score rows can use `ScoreProfile`
without implementing `MotifModel`:

```python
from mimosa import RaggedArray, ScoreProfile, compare, prepare_profile

external = ScoreProfile(
    "external",
    RaggedArray.from_rows([[1.0, 2.0], [3.0]]),
)
prepared = prepare_profile(external)
```

`ScoreProfile` supports preparation and comparison, but not raw-sequence
scanning or motif-site extraction. A precomputed profile alone cannot support
input relabeling: the original function must be evaluable on recoded windows.
