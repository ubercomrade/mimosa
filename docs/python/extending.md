# Custom Models and Readers

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
`select_sites`, and `reconstruct_pfm`. A stable SHA-256 `fingerprint()` is
needed only when using the prepared-profile cache. Arbitrary custom models are
not accepted by `write_model` because `scan_into` does not define how to
serialize model parameters.

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
scanning or motif-site extraction.

## Custom model readers

Pass readers to one `read_model` call. The reader must expose `formats`,
`probe(path)`, and `read(path, **kwargs)`:

```python
from mimosa import read_model


class CustomReader:
    formats = ("custom",)

    def probe(self, path):
        return path.suffix == ".custom"

    def read(self, path, **kwargs):
        return load_custom_model(path)


model = read_model("model.custom", readers=(CustomReader(),))
```

Automatic detection uses the file extension when it identifies one reader and
calls `probe` when several candidates remain. There is no mutable global reader
registry.
