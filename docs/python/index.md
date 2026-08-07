# Python Documentation

Mimosa is a Python library and CLI for comparing DNA motif models through the
score profiles they produce on the same sequences.

## Guides

- [Quickstart](quickstart.md) - install Mimosa and run the main workflows.
- [API](api.md) - public objects, functions, and result types.
- [CLI](cli.md) - commands, options, output, and exit behavior.
- [Models and formats](models.md) - built-in models and readers.
- [Method and statistics](method.md) - normalization, alignment, metrics, and nulls.
- [Custom models and readers](extending.md) - extend Mimosa without private imports.
- [Storage and cache](storage.md) - portable bundles and prepared-profile cache.
- [Data layout](data_layout.md) - encoding, ragged arrays, geometry, and coordinates.

The root package exports the stable workflow API. Examples in these documents
use public imports unless a lower-level I/O utility is explicitly documented.
