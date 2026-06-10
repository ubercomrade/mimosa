# План улучшения качества кода, tooling, архитектуры и hygiene проекта MIMOSA

> Этот план составлен по результатам инспекции текущего проекта `mimosa`.
> Команды в разделе validation checklist нужно выполнить отдельно; в рамках ревью они не запускались.

## 0. Статус выполнения

Статус обновлён после текущего refactoring pass.

### Выполнено

- P0.1 — package metadata lookup использует distribution name `mimosa-tool`.
- P0.2 — benchmark scripts больше не игнорируются и приведены к Ruff formatting.
- P0.3 — `uv.lock` больше не игнорируется; lockfile обновлён после dev-dependency changes.
- P0.4 — `.hocomoco` исключён из tooling analysis через config/ignore.
- P0.5 — README и code docs отмечают trusted-only `.pkl`/`.joblib` loading.
- P1.1 — NumPy scalar constants в `batches.py` заменены на Python scalars.
- P1.2 — XML required helpers добавлены и используются в SLIM/Dimont parsing.
- P1.3 — XML array parsing типизирован через `XmlArray` и required helper.
- P1.4 — добавлен `_copy_comparator_config()`.
- P1.6 — добавлены runtime invariant checks для threshold flow.
- P1.7 — добавлен мягкий `mypy` config; `uv run mypy src/mimosa` проходит.
- P2.1 — CI matrix расширена до Python `3.10`–`3.13`; classifier `3.14` убран.
- P2.2 — добавлен `pytest-cov`, coverage config и CI coverage command.
- P2.3 — добавлен `pyscn`; `uv run pyscn analyze .` проверен как advisory command.
- P2.4 — `MANIFEST.in` заменён на whitelist examples; sdist content проверен.
- P2.5 — publish workflow валидирует `uv build`, `twine check`, install-from-sdist, smoke import и tests до upload/publish.
- P3.6 — CLI больше не импортирует private `_validate_metric`; добавлен public `validate_metric()`.
- P3.3 — model registry типизирован через `ModelHandler`.
- P3.7 — добавлены `TypedDict`-схемы для null artifacts.
- P3.9 — `validate_metric()` экспортирован из top-level API и покрыт тестом.
- P3.8 — усилен malformed input handling для MEME, PFM, SiteGA и BaMM; добавлены negative tests.
- P3.2 — `io.py` разбит на тематические подмодули с сохранением фасада `mimosa.io`.
- P3.4 — `tests/test_unit.py` разбит на тематические unit-модули с совместимым агрегатором.

### Частично выполнено

- P1.5 — уточнены return types для comparison/API/profile bundle/cache, но candidate/null/model typing ещё можно развивать.
- P3.1 — выбран путь B (`TypedDict` + helpers); полноценный dataclass transition не выполнялся.
- P3.5 — integration assertions усилены общим JSON invariant helper; stable golden checks ещё не добавлялись.

### Не выполнено

- Нет.

## 1. Цель

Привести проект к более устойчивому состоянию без потери текущей функциональности и производительности:

- убрать высокосигнальные static/type diagnostics;
- улучшить tooling: lint, format, type checking, coverage, structural analysis;
- устранить противоречия в `.gitignore`, CI и packaging;
- повысить безопасность загрузки пользовательских артефактов;
- сохранить производительность NumPy/Numba kernels;
- сделать архитектуру проще для поддержки и расширения;
- сделать CI/release workflow более воспроизводимым.

## 2. Текущее состояние

### 2.1. Сильные стороны

Проект уже имеет хорошую базу:

- используется `src` layout: `src/mimosa`;
- логичное разделение модулей:
  - `api.py` — публичный API;
  - `cli.py` — CLI shell;
  - `comparison.py` — orchestration comparison workflows;
  - `functions.py` — численные функции и Numba kernels;
  - `models.py` — model registry и scan/read/write dispatch;
  - `io.py` — чтение/запись форматов;
  - `batches.py`, `cache.py`, `validation.py`, `nulls.py` — отдельные support-модули;
- много unit и integration tests;
- есть CI с `ruff check`, `ruff format --check`, `pytest`;
- есть кэширование derived profiles;
- есть явная ориентационная логика и Numba-оптимизации для profile scoring.

### 2.2. Основные проблемы

1. Static/type diagnostics не чистые:
   - `src/mimosa/io.py`;
   - `src/mimosa/comparison.py`;
   - `src/mimosa/models.py`;
   - `src/mimosa/functions.py`;
   - `src/mimosa/batches.py`;
   - `src/mimosa/nulls.py`;
   - tests/examples в IDE-окружении.

2. Tooling неполный:
   - в `dev` dependency group только `pytest` и `ruff`;
   - нет `mypy`/`pyright`;
   - нет coverage;
   - нет `pyscn` или аналогичного structural analysis.

3. CI не соответствует заявленной поддержке Python:
   - classifiers заявляют Python `3.10`–`3.14`;
   - CI тестирует только Python `3.12`.

4. Hygiene-противоречия:
   - `.gitignore` игнорирует `uv.lock`, но проект использует `uv sync`;
   - `.gitignore` игнорирует benchmark scripts, но CI их запускает;
   - `.hocomoco` игнорируется, но лежит внутри workspace и даёт diagnostics;
   - `MANIFEST.in` включает весь `examples/*`.

5. Возможный metadata bug:
   - distribution name в `pyproject.toml`: `mimosa-tool`;
   - `src/mimosa/nulls.py` вызывает `package_metadata.version("mimosa")`.

6. Security:
   - `joblib.load()` используется для `.pkl` моделей и null artifacts;
   - нужно явно документировать, что такие файлы должны быть trusted-only.

## 3. Принципы изменений

1. **Не ломать поведение без необходимости.**
   Все изменения в numerical logic должны сопровождаться тестами.

2. **Сначала high-signal cleanup, потом архитектурные переработки.**
   Не начинать большой refactor до устранения metadata/tooling/hygiene проблем.

3. **Типизацию вводить постепенно.**
   Для NumPy/Numba проекта strict type checking сразу будет слишком шумным.

4. **Сохранять performance-critical code простым.**
   Не вводить тяжёлые абстракции вокруг Numba kernels.

5. **Делать I/O ошибки доменными и понятными.**
   Malformed MEME/XML/SiteGA/BaMM inputs должны давать `ValueError` с контекстом, а не случайный `TypeError`.

6. **Разделять trusted и untrusted inputs.**
   Pickle/joblib artifacts — только trusted.

## 4. Приоритет P0 — быстрые обязательные исправления

### ✅ P0.1. Исправить package metadata lookup

**Файл:** `src/mimosa/nulls.py`

Сейчас:

```python
version = package_metadata.version("mimosa")
```

Нужно:

```python
version = package_metadata.version("mimosa-tool")
```

**Зачем:** distribution name в `pyproject.toml` — `mimosa-tool`; иначе metadata null artifacts может всегда получать `"0+unknown"`.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k null
```

Добавить/обновить тест, который monkeypatch-ит `importlib.metadata.version` и проверяет, что вызывается `"mimosa-tool"`.

---

### ✅ P0.2. Устранить конфликт `.gitignore` и CI по benchmark scripts

**Файлы:**

- `.gitignore`
- `.github/workflows/test.yml`
- `examples/benchmark_fast_profile_score.py`
- `examples/benchmark_profile_one_to_many.py`

Сейчас `.gitignore` содержит:

```gitignore
examples/benchmark_fast_profile_score.py
examples/benchmark_profile_one_to_many.py
```

Но CI запускает эти файлы.

Нужно выбрать одно решение:

#### Вариант A — benchmark scripts являются частью проекта

- убрать benchmark scripts из `.gitignore`;
- оставить CI smoke-run;
- привести benchmark scripts к Ruff formatting.

#### Вариант B — benchmark scripts локальные

- убрать их запуск из CI;
- оставить в `.gitignore`;
- перенести в `scripts/local/` или оставить локально вне repo.

**Рекомендуемый вариант:** A, потому что CI уже использует эти scripts как smoke tests.

**Проверить:**

```bash
uv run ruff check examples
uv run ruff format --check examples
```

---

### ✅ P0.3. Решить судьбу `uv.lock`

**Файлы:**

- `.gitignore`
- `uv.lock`

Если проект использует `uv sync --group dev`, lockfile лучше хранить в git для воспроизводимости.

Нужно:

- убрать из `.gitignore` строки:
  ```gitignore
  ./uv.lock
  uv.lock
  .uv.lock
  *.lock
  ```
  или хотя бы не игнорировать `uv.lock`;
- убедиться, что `uv.lock` актуален.

**Проверить:**

```bash
uv sync --group dev
```

---

### ✅ P0.4. Убрать `.hocomoco` из анализа и рабочего дерева проекта

**Файлы/папки:**

- `.hocomoco/`
- `.gitignore`
- возможно настройки IDE/pyright/ruff excludes

`.hocomoco/examples.py` содержит локальный экспериментальный код:

- hardcoded absolute path;
- отсутствующий импорт `tqdm`;
- устаревший параметр `n_permutations`;
- потенциально неверный `fetch_family()`.

Нужно:

- не держать `.hocomoco` внутри workspace проекта;
- либо явно исключить из анализа;
- не включать в CI/lint/test.

Если добавляется `pyrightconfig.json`, включить:

```json
{
  "exclude": [
    ".hocomoco",
    ".venv",
    "dist",
    "build"
  ]
}
```

Если используется Ruff exclude:

```toml
[tool.ruff]
extend-exclude = [".hocomoco", "dist", "build", ".venv"]
```

**Проверить:** IDE diagnostics больше не должны показывать `.hocomoco`.

---

### ✅ P0.5. Добавить security warning для pickle/joblib loading

**Файлы:**

- `README.md`
- `src/mimosa/models.py`
- `src/mimosa/nulls.py`
- возможно CLI help в `src/mimosa/cli.py`

Нужно документировать:

- `.pkl` model files загружаются через `joblib.load`;
- null artifacts загружаются через `joblib.load`;
- эти файлы нельзя загружать из недоверенных источников.

Минимальный README-блок:

```markdown
### Security note

MIMOSA can load `.pkl` model files and `.joblib` null-distribution artifacts.
These formats use Python pickle/joblib serialization and may execute arbitrary
code when loaded. Only load such files from trusted sources.
```

**Проверить:**

```bash
uv run pytest
```

---

## 5. Приоритет P1 — убрать high-signal diagnostics и улучшить типизацию

### ✅ P1.1. Исправить NumPy scalar constants в `batches.py`

**Файл:** `src/mimosa/batches.py`

Сейчас:

```python
NUCLEOTIDE_PADDING = np.int8(4)
SCORE_PADDING = np.float32(0.0)
```

Лучше:

```python
NUCLEOTIDE_PADDING = 4
SCORE_PADDING = 0.0
```

А dtype задавать при создании массивов:

```python
np.full(..., NUCLEOTIDE_PADDING, dtype=np.int8)
np.full(..., SCORE_PADDING, dtype=np.float32)
```

**Зачем:** меньше конфликтов с `TypedDict` и type checkers.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k batch
uv run pytest tests/test_unit.py -k profile
```

---

### ✅ P1.2. Добавить required XML helpers в `io.py`

**Файл:** `src/mimosa/io.py`

Проблема: `_xml_numeric_value()` возвращает `float | None`, но код часто вызывает:

```python
int(_xml_numeric_value(...))
```

Нужно добавить helpers:

```python
def _required_xml_numeric(elem: ET.Element | None, label: str) -> float:
    value = _xml_numeric_value(elem)
    if value is None:
        raise ValueError(f"Malformed XML: missing numeric value for {label}")
    return value


def _required_xml_child(parent: ET.Element, path: str, label: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise ValueError(f"Malformed XML: missing {label}")
    return element
```

И заменить опасные места в:

- `_parse_slim_model`;
- `_parse_dimont_treeelement`;
- `_parse_dimont_model`.

**Зачем:** убрать type diagnostics и сделать ошибки парсинга понятнее.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k "dimont or slim"
```

Добавить negative tests для malformed XML.

---

### ✅ P1.3. Типизировать XML array parsing

**Файл:** `src/mimosa/io.py`

Сейчас `_xml_array()` рекурсивно возвращает гетерогенный тип:

```python
float | list[...] | None
```

Нужно:

- либо явно типизировать alias:
  ```python
  XmlArray = float | list["XmlArray"]
  ```
- либо разделить:
  - `_xml_array_optional`;
  - `_required_xml_array`.

Пример:

```python
XmlArray = float | list["XmlArray"]

def _required_xml_array(elem: ET.Element | None, label: str) -> XmlArray:
    if elem is None:
        raise ValueError(f"Malformed XML: missing {label}")
    value = _xml_array(elem)
    if value is None:
        raise ValueError(f"Malformed XML: empty {label}")
    return value
```

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k slim
```

---

### ✅ P1.4. Сделать copying `ComparatorConfig` типобезопасным

**Файл:** `src/mimosa/comparison.py`

Проблема:

```python
effective_config = dict(config)
```

Type checker видит `dict[str, object]`, а не `ComparatorConfig`.

Нужно добавить helper:

```python
from typing import cast

def _copy_comparator_config(config: ComparatorConfig) -> ComparatorConfig:
    return cast(ComparatorConfig, dict(config))
```

И заменить:

```python
effective_config = dict(config)
```

на:

```python
effective_config = _copy_comparator_config(config)
```

**Зачем:** убрать cascade diagnostics в `compare()` и `compare_one_to_many()`.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k comparator
uv run pytest tests/test_unit.py -k one_to_many
```

---

### 🟡 P1.5. Уточнить return types в comparison/model helpers

**Файлы:**

- `src/mimosa/comparison.py`
- `src/mimosa/models.py`
- `src/mimosa/functions.py`

Нужно постепенно заменить broad `dict` на более точные типы:

- `ComparisonResult`;
- `ProfileBundle`;
- `MaskedBatch`;
- `SequenceBatch`;
- локальные `TypedDict` для candidates:
  - profile candidate;
  - motif candidate;
  - null artifact entry.

Пример для motif candidate:

```python
class MotifCandidate(TypedDict):
    orientation: str
    score: float
    offset: int
```

**Зачем:** уменьшить `Any`/`dict` propagation.

**Проверить:**

```bash
uv run pytest
```

---

### ✅ P1.6. Явно проверять runtime invariants в `models.py`

**Файл:** `src/mimosa/models.py`

Места:

- `_collect_hits`;
- `_resolve_hits`;
- `_extract_site_matrix`;
- threshold table handling.

Например:

```python
if mode == "threshold":
    if score_threshold is None:
        raise ValueError("score_threshold is required in threshold mode")
    return _collect_threshold_hits(score_bundle, score_threshold, strand)
```

И:

```python
if mode == "threshold":
    if threshold_table is None:
        raise ValueError("threshold_table is required in threshold mode")
```

**Зачем:** type checker видит инварианты, runtime errors становятся понятнее.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k "threshold or get_sites or get_pfm"
```

---

### ✅ P1.7. Добавить мягкую mypy/pyright конфигурацию

**Файл:** `pyproject.toml` или `pyrightconfig.json`

Начать не со strict mode.

Вариант для `mypy`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.3.0",
    "mypy>=1.10",
    "pyscn>=0.1",
]

[tool.mypy]
python_version = "3.10"
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
ignore_missing_imports = true
```

Первый этап можно проверять только package code:

```bash
uv run mypy src/mimosa
```

Tests/examples подключать позже.

---

## 6. Приоритет P2 — tooling, CI и packaging hygiene

### ✅ P2.1. Расширить CI matrix или сузить classifiers

**Файлы:**

- `pyproject.toml`
- `.github/workflows/test.yml`

Сейчас classifiers заявляют Python `3.10`–`3.14`, но CI тестирует только `3.12`.

Нужно выбрать:

#### Вариант A — реально поддерживать несколько версий

```yaml
python-version: ["3.10", "3.11", "3.12", "3.13"]
```

Python `3.14` добавлять только если зависимости, особенно `numba`, его поддерживают.

#### Вариант B — сузить classifiers

Оставить только версии, которые реально тестируются.

**Рекомендуемый вариант:** matrix `3.10`–`3.13`, `3.14` только после проверки Numba compatibility.

**Проверить:** CI green на всех версиях.

---

### ✅ P2.2. Добавить coverage

**Файл:** `pyproject.toml`

Добавить dependency:

```toml
"pytest-cov>=5.0"
```

Добавить команду в CI:

```bash
uv run pytest --cov=src/mimosa --cov-report=term-missing
```

Не ставить высокий threshold сразу. Начать с фиксации baseline:

```toml
[tool.coverage.run]
source = ["src/mimosa"]

[tool.coverage.report]
show_missing = true
```

Позже добавить threshold.

---

### ✅ P2.3. Добавить structural analysis

**Файл:** `pyproject.toml`

Добавить `pyscn` или аналог.

Цели:

- dead code;
- duplicate code;
- complexity;
- circular dependencies.

Начать как advisory command, не как hard CI gate:

```bash
uv run pyscn analyze .
```

Позже добавить:

```bash
uv run pyscn check .
```

---

### ✅ P2.4. Уточнить `MANIFEST.in`

**Файл:** `MANIFEST.in`

Сейчас:

```ini
recursive-include examples *
```

Лучше заменить на whitelist:

```ini
recursive-include examples *.fa *.fasta *.meme *.pfm *.mat *.ihbcp *.hbcp *.sh *.ps1
```

И исключить generated outputs:

```ini
global-exclude __pycache__
global-exclude *.py[cod]
global-exclude .DS_Store
global-exclude .pytest_cache
global-exclude .ruff_cache
global-exclude .mimosa-cache
```

Если benchmark scripts должны входить в sdist, добавить явно:

```ini
include examples/benchmark_fast_profile_score.py
include examples/benchmark_profile_one_to_many.py
```

**Проверить:**

```bash
uv build
uvx twine check dist/*
```

---

### ✅ P2.5. Проверить package installation from sdist/wheel

**Файлы:**

- `.github/workflows/publish.yml`
- `pyproject.toml`
- `MANIFEST.in`

Добавить release validation:

```bash
uv build
uvx twine check dist/*
uv venv test_env
. test_env/bin/activate
uv pip install dist/*.tar.gz
python -c "import mimosa; print(mimosa.__all__)"
pytest tests
```

---

## 7. Приоритет P3 — архитектурная чистка

### 🟡 P3.1. Рассмотреть переход configs с `TypedDict` на dataclass

**Файлы:**

- `src/mimosa/comparison.py`
- `src/mimosa/api.py`
- `src/mimosa/cli.py`

Сейчас configs — mutable dict-like `TypedDict`. Это удобно для API compatibility, но порождает type issues.

Варианты:

#### Вариант A — оставить public dict API, внутри использовать dataclass

```python
@dataclass(frozen=True)
class ComparatorOptions:
    metric: MetricName
    seed: int | None = None
    ...
```

`create_comparator_config()` может временно возвращать `ComparatorConfig`, но внутренние функции принимают dataclass.

#### Вариант B — оставить `TypedDict`, но ввести helpers

- `_copy_comparator_config`;
- `_validate_comparator_config`;
- typed accessors.

**Рекомендуемый путь:** сначала B, затем A только если type debt останется высоким.

---

### ✅ P3.2. Разделить `io.py` на подмодули

**Текущий файл:** `src/mimosa/io.py`

Файл содержит разные форматы:

- FASTA;
- scores;
- MEME/PFM;
- SiteGA;
- BaMM;
- SLIM XML;
- Dimont XML;
- writers.

Можно постепенно разделить:

```text
src/mimosa/io/
  __init__.py
  fasta.py
  scores.py
  meme.py
  sitega.py
  bamm.py
  jstacs.py
  writers.py
```

Но это breaking refactor для imports. Поэтому делать осторожно:

1. сначала добавить private helper sections внутри `io.py`;
2. затем вынести без изменения public imports;
3. в `src/mimosa/io/__init__.py` re-export старые функции.

---

### ✅ P3.3. Типизировать model registry через Protocol

**Файл:** `src/mimosa/models.py`

Сейчас:

```python
registry: Dict[str, dict] = {}
```

Лучше ввести `TypedDict` или `Protocol` для handler bundle:

```python
class ModelHandler(TypedDict):
    scan: Callable[..., MaskedBatch]
    scan_both: Callable[..., tuple[MaskedBatch, MaskedBatch]] | None
    load: Callable[..., GenericModel]
    write: Callable[..., None]
    score_bounds: Callable[..., tuple[float, float]]
```

**Зачем:** уменьшить `Any`

<thinking>
**Planning execution phases**

I see we have tasks outlined, but I’m considering including a matrix for clarity. The execution order seems to be structured into phases: Phase 0 is about taking a snapshot with no behavior, Phase 1 focuses on metadata and hygiene, Phase 2 handles type diagnostics, Phase 3 is all about tooling, Phase 4 covers tests and architecture, and finally, Phase 5 is the release. I’m curious if this sequence will be the most efficient approach!
</thinking>

Продолжаю текст `PLAN.md` с места, где он оборвался:

```markdown
**Зачем:** уменьшить `Any`, ловить ошибки регистрации model handlers раньше и сделать dispatch в `models.py` понятнее для type checker.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k "model_registry or scan_model or read_model"
```

---

### ✅ P3.4. Разбить монолитный `tests/test_unit.py`

**Файл сейчас:** `tests/test_unit.py`

Сейчас файл очень большой и содержит тесты для разных уровней:

- numerical functions;
- IO parsers;
- model registry;
- comparison strategies;
- API;
- cache;
- null distributions.

Постепенно разделить на:

```text
tests/
  test_batches.py
  test_functions.py
  test_io.py
  test_models.py
  test_comparison_profile.py
  test_comparison_motif.py
  test_api.py
  test_cache.py
  test_nulls.py
  test_cli_config.py
  test_integration.py
```

**Порядок:**

1. Сначала переносить тесты без изменения логики.
2. Общие helpers вынести в `tests/conftest.py` или `tests/helpers.py`.
3. После каждого небольшого переноса запускать релевантные тесты.

**Проверить:**

```bash
uv run pytest tests
```

---

### 🟡 P3.5. Усилить integration tests

**Файл:** `tests/test_integration.py`

Сейчас многие integration tests проверяют только наличие ключей JSON.

Добавить проверки инвариантов:

```python
assert output["metric"] == "co"
assert output["orientation"] in {"++", "+-", "-+", "--"}
assert isinstance(output["offset"], int)
assert np.isfinite(output["score"])
```

Для метрик `co`, `dice`, `cosine`:

```python
assert 0.0 <= output["score"] <= 1.0
```

Для `ed` score может быть отрицательным, поэтому диапазон должен быть отдельным.

Для стабильных example fixtures можно добавить approximate golden checks:

```python
assert output["offset"] == expected_offset
assert output["score"] == pytest.approx(expected_score, rel=1e-6)
```

**Важно:** golden checks добавлять только там, где результат стабилен между Python/NumPy/Numba версиями.

**Проверить:**

```bash
uv run pytest tests/test_integration.py
```

---

### ✅ P3.6. Убрать private import из CLI

**Файл:** `src/mimosa/cli.py`

Сейчас CLI импортирует private function:

```python
from mimosa.comparison import _validate_metric
```

Лучше сделать одно из:

#### Вариант A — сделать функцию публичной

В `comparison.py`:

```python
def validate_metric(metric: str) -> MetricName:
    ...
```

И внутри оставить compatibility alias, если нужно:

```python
_validate_metric = validate_metric
```

В `cli.py` использовать:

```python
from mimosa.comparison import validate_metric
```

#### Вариант B — не использовать validation helper в CLI

В `cli.py` использовать уже существующие `SUPPORTED_PROFILE_METRICS` / `SUPPORTED_MOTIF_METRICS` и strategy-specific validation.

**Рекомендуемый вариант:** A, потому что validation уже централизована.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k "comparator_config or cli_maps"
uv run ruff check src/mimosa/cli.py src/mimosa/comparison.py
```

---

### ✅ P3.7. Описать схемы result/null artifacts

**Файлы:**

- `src/mimosa/comparison.py`
- `src/mimosa/nulls.py`

Сейчас result и artifact entries — в основном `dict[str, Any]`.

Добавить `TypedDict`:

```python
class NullArtifactMetadata(TypedDict):
    format_version: int
    created_at: str
    strategy: str
    metric: str
    config_signature: dict[str, Any]
    config_signature_hash: str
    sequence_fingerprint: str
    background_fingerprint: str
    model_collection_fingerprint: str | None
    relation_fingerprint: str | None
    package_version: str


class NullArtifactEntry(TypedDict, total=False):
    estimator_type: str
    sorted_scores: np.ndarray
    parameters: dict[str, Any]
    query_name: str
    query_fingerprint: str
    included_target_names: list[str]
    included_target_fingerprints: list[str]
    effective_number_of_targets: int
    raw_null_scores: np.ndarray
    n_null: int
```

**Зачем:** уменьшить риск ошибок в artifact compatibility и annotation code.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k null
```

---

### ✅ P3.8. Улучшить error handling в IO readers

**Файл:** `src/mimosa/io.py`

Добавить более явную валидацию:

#### MEME

В `read_meme()` сейчас `parts[1]` используется без проверки. Нужно как в `read_meme_many()`:

```python
if len(parts) < _MEME_MIN_MOTIF_FIELDS:
    raise ValueError(f"Malformed MEME file {path}: MOTIF line has no name.")
```

Также проверить:

- `length <= 0`;
- количество считанных matrix rows;
- matrix shape.

#### PFM

В `read_pfm()`:

```python
pfm = np.loadtxt(path, comments=">").T
```

Добавить проверки:

- matrix is 2D;
- one axis has size 4 or 5;
- values finite;
- length > 0.

#### SiteGA

В `read_sitega()` добавить проверки:

- header не пустой;
- length positive;
- строки имеют 5 полей;
- dinucleotide валидный;
- start/stop внутри диапазона.

#### BaMM

В `parse_file_content()` добавить проверки:

- все строки order имеют ожидаемую длину;
- probabilities finite;
- motif_length > 0.

**Проверить:**

```bash
uv run pytest tests/test_unit.py -k "read_meme or read_pfm or sitega or bamm"
```

Добавить negative tests для malformed inputs.

---

### 🟡 P3.9. Явно отделить public API от internal API

**Файлы:**

- `src/mimosa/__init__.py`
- `src/mimosa/api.py`
- `src/mimosa/comparison.py`
- `src/mimosa/models.py`

Проверить, что в `__all__` экспортируются только стабильные public functions/classes.

Internal helpers с `_` не должны использоваться в examples/tests, кроме тестов internal behavior. Если tests проверяют private helpers, делать это осознанно и не рекламировать их как public API.

**Проверить:**

```bash
python -c "import mimosa; print(sorted(mimosa.__all__))"
```

---

## 8. Конкретные задачи по файлам

### `pyproject.toml`

Сделать:

1. Добавить dev dependencies:
   - `mypy`;
   - `pytest-cov`;
   - `pyscn` или выбранный аналог.
2. Добавить мягкую mypy-конфигурацию.
3. Добавить coverage-конфигурацию.
4. Проверить classifiers против CI matrix.
5. При необходимости добавить `tool.ruff.extend-exclude`.

Проверить:

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run mypy src/mimosa
```

---

### `.github/workflows/test.yml`

Сделать:

1. Расширить Python matrix или сузить classifiers.
2. Добавить type check step, сначала non-blocking или только `src/mimosa`.
3. Добавить coverage step.
4. Убедиться, что examples/benchmarks, которые запускаются в CI, tracked и не игнорируются.

Пример будущего workflow:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest]
    python-version: ["3.10", "3.11", "3.12", "3.13"]

steps:
  - uses: actions/checkout@v6
  - uses: astral-sh/setup-uv@v7
    with:
      python-version: ${{ matrix.python-version }}
  - run: uv sync --group dev
  - run: uv run ruff check .
  - run: uv run ruff format --check .
  - run: uv run mypy src/mimosa
  - run: uv run pytest --cov=src/mimosa --cov-report=term-missing
```

---

### `.github/workflows/publish.yml`

Сделать:

1. Перед publish запускать:
   - `uv build`;
   - `twine check`;
   - install from sdist;
   - import smoke test;
   - pytest from installed package.
2. Убедиться, что `uv publish` не имеет лишнего пробела и настроен через trusted publishing.

---

### `.gitignore`

Сделать:

1. Решить судьбу `uv.lock`.
2. Убрать benchmark scripts из ignore, если они часть проекта.
3. Оставить `.hocomoco` ignored.
4. Убедиться, что generated outputs/caches ignored:
   - `.mimosa-cache/`;
   - `.pytest_cache/`;
   - `.ruff_cache/`;
   - `.mypy_cache/`;
   - `dist/`;
   - `build/`;
   - `*.egg-info/`.

---

### `MANIFEST.in`

Сделать:

1. Заменить broad include examples на whitelist.
2. Исключить generated outputs.
3. Проверить sdist content.

Проверить:

```bash
uv build
tar -tf dist/*.tar.gz | sort | less
```

---

### `src/mimosa/nulls.py`

Сделать:

1. Исправить package metadata distribution name.
2. Добавить typing для artifact metadata/entries.
3. Документировать trusted-only `joblib.load`.
4. Возможно добавить basic validation artifact schema после загрузки:
   - `metadata` exists;
   - `entries` exists;
   - `format_version` supported.

Проверить:

```bash
uv run pytest tests/test_unit.py -k null
```

---

### `src/mimosa/batches.py`

Сделать:

1. Python scalar constants вместо NumPy scalar constants.
2. Более точные type hints для functions:
   - `make_sequence_batch`;
   - `make_score_batch`;
   - `make_strand_bundle`;
   - `profile_view`;
   - `flatten_profile_bundle`.

Проверить:

```bash
uv run pytest tests/test_unit.py -k "batch or profile_bundle"
```

---

### `src/mimosa/io.py`

Сделать:

1. Required XML numeric/child helpers.
2. Typed XML array helpers.
3. Улучшить malformed input errors.
4. Добавить/обновить negative tests.
5. Рассмотреть future split на format-specific modules.

Проверить:

```bash
uv run pytest tests/test_unit.py -k "read_model or dimont or slim or meme or pfm or sitega or bamm"
```

---

### `src/mimosa/comparison.py`

Сделать:

1. `_copy_comparator_config`.
2. Public `validate_metric`.
3. Более точные candidate/result types.
4. Уменьшить broad `dict`.
5. Проверить one-to-many parallel path с typed config.
6. Сохранить performance-critical loops без лишней абстракции.

Проверить:

```bash
uv run pytest tests/test_unit.py -k "comparison or profile or motif or one_to_many"
```

---

### `src/mimosa/models.py`

Сделать:

1. Typed model handler registry.
2. Явные runtime invariant checks.
3. Улучшить type hints для scan functions.
4. Документировать pickle-loading risk.
5. Проверить threshold table flow.

Проверить:

```bash
uv run pytest tests/test_unit.py -k "model or scan or get_sites or get_pfm or threshold"
```

---

### `src/mimosa/functions.py`

Сделать:

1. Уточнить return type `prepare_profile_bundle()`.
2. Убедиться, что Numba-call signatures не конфликтуют с Python scalar constants.
3. Не переписывать kernels без benchmark/regression tests.
4. Добавить small tests для edge cases:
   - empty arrays;
   - all masked rows;
   - zero denominator metrics.

Проверить:

```bash
uv run pytest tests/test_unit.py -k "functions or scores or rowwise or curve"
```

---

### `src/mimosa/cli.py`

Сделать:

1. Убрать private import `_validate_metric`.
2. Добавить warning/help для trusted `.joblib` artifacts.
3. Проверить `build-null` validation.
4. Возможно вынести CLI args mapping в отдельный typed helper.

Проверить:

```bash
uv run pytest tests/test_unit.py -k cli
uv run pytest tests/test_integration.py
```

---

### `tests/`

Сделать:

1. Постепенно разбить `test_unit.py`.
2. Усилить integration assertions.
3. Добавить malformed input tests.
4. Добавить tests для metadata package name.
5. Добавить tests для security messaging, если CLI help меняется.

---

### `examples/`

Сделать:

1. Решить, какие examples являются shipped examples.
2. Убрать generated outputs из package data.
3. Убедиться, что `run.sh` работает в CI.
4. Проверить benchmark scripts Ruff formatting.
5. Если benchmark scripts остаются, не игнорировать их.

---

## 9. Порядок выполнения

### ✅ Фаза 0 — подготовка

1. Зафиксировать текущий baseline:
   ```bash
   uv sync --group dev
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```
2. Не исправлять всё сразу. Сначала small PR/change set.

---

### ✅ Фаза 1 — P0 hygiene fixes

1. `package_metadata.version("mimosa-tool")`.
2. `.gitignore` vs benchmark scripts.
3. `uv.lock` policy.
4. `.hocomoco` exclusion/removal.
5. Security warning in README.

Проверить:

```bash
uv run ruff check .
uv run pytest
```

---

### ✅ Фаза 2 — high-signal typing fixes

1. `batches.py` scalar constants.
2. XML required helpers in `io.py`.
3. `_copy_comparator_config` in `comparison.py`.
4. Runtime invariant asserts/checks in `models.py`.

Проверить:

```bash
uv run pytest tests/test_unit.py
uv run pytest tests/test_integration.py
```

---

### ✅ Фаза 3 — tooling

1. Add `mypy`, `pytest-cov`, `pyscn`.
2. Add soft mypy config.
3. Add coverage config.
4. Update CI.

Проверить:

```bash
uv sync --group dev
uv run mypy src/mimosa
uv run pytest --cov=src/mimosa --cov-report=term-missing
```

---

### 🟡 Фаза 4 — tests and architecture cleanup

1. Split `tests/test_unit.py`.
2. Strengthen integration assertions.
3. Add malformed input tests.
4. Add typed registry.
5. Reduce `dict[str, Any]` in null artifacts/results.

Проверить:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

---

### 🟡 Фаза 5 — packaging/release hygiene

1. Update `MANIFEST.in`.
2. Build sdist/wheel.
3. Install from sdist.
4. Run smoke tests from installed package.
5. Update publish workflow.

Проверить:

```bash
uv build
uvx twine check dist/*
```

---

## 10. Validation checklist

Перед merge/release выполнить:

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests/test_unit.py
uv run pytest tests/test_integration.py
```

После добавления mypy:

```bash
uv run mypy src/mimosa
```

После добавления coverage:

```bash
uv run pytest --cov=src/mimosa --cov-report=term-missing
```

После добавления structural analysis:

```bash
uv run pyscn analyze .
```

Перед release:

```bash
uv build
uvx twine check dist/*
```

Для CLI smoke test:

```bash
uv run mimosa --help
uv run mimosa profile examples/scores_1.fasta examples/scores_2.fasta \
  --model1-type scores \
  --model2-type scores \
  --metric cosine
```

---

## 11. Definition of done

Проект считается приведённым в хорошее состояние, когда:

1. `ruff check .` проходит без ошибок.
2. `ruff format --check .` проходит без ошибок.
3. `pytest` проходит полностью.
4. IDE diagnostics по `src/mimosa` не показывают high-signal ошибок.
5. `mypy src/mimosa` либо проходит, либо имеет минимальный documented baseline.
6. CI matrix соответствует заявленным Python classifiers.
7. `.gitignore` не противоречит CI.
8. `MANIFEST.in` не включает случайные generated outputs.
9. `README.md` содержит предупреждение про trusted `.pkl`/`.joblib`.
10. Null artifact metadata использует правильное distribution name `mimosa-tool`.
11. XML/MEME/PFM/SiteGA malformed inputs дают понятные `ValueError`.
12. `tests/test_unit.py` разбит на тематические helper/test-модули с сохранением старого entry point.
13. Release build проходит `twine check`.

---

## 12. Что не делать без отдельного решения

1. Не переписывать Numba kernels ради красоты без benchmark evidence.
2. Не включать strict mypy сразу на весь проект.
3. Не менять публичный API без compatibility layer.
4. Не удалять поддержку `.pkl`/`.joblib` без migration path.
5. Не менять численные формулы без regression tests.
6. Не добавлять новые heavy dependencies без явной пользы.
7. Не смешивать formatting/tooling cleanup с behavior changes в одном большом change set.

---

## 13. Рекомендуемый первый маленький change set

Самый безопасный первый набор изменений:

1. `src/mimosa/nulls.py`:
   - `version("mimosa")` → `version("mimosa-tool")`.

2. `.gitignore`:
   - убрать ignore benchmark scripts, если они остаются в CI;
   - решить судьбу `uv.lock`.

3. `README.md`:
   - добавить security note про pickle/joblib.

4. `pyproject.toml`:
   - добавить `tool.ruff.extend-exclude = [".hocomoco"]`, если `.hocomoco` остаётся внутри workspace.

5. Запустить:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```

После этого переходить к `batches.py` и `io.py`.
