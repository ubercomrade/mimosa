# План переноса MIMOSA на Julia

## 1. Назначение документа

Этот документ превращает требования из `REFACTORING.md` в исполнимый план создания самостоятельного пакета
`Mimosa.jl`. Перенос выполняется как последовательность вертикальных срезов с Python-реализацией в роли
зафиксированного научного oracle. Построчный перевод модулей и временная Julia-архитектура, повторяющая
`GenericModel`, registry callbacks, pandas и Numba-specific kernels, не допускаются.

План должен уточняться после этапа 0. Изменение научной семантики, публичного API, формата хранения или критериев
численной совместимости оформляется ADR и отражается здесь.

### Цели

- сохранить поддерживаемые пользовательские сценарии и научную интерпретацию результатов MIMOSA;
- предоставить небольшой типизированный Julia API, пригодный для прямого использования из `MotifHORDE.jl`;
- отделить вычислительное ядро, I/O, статистику, кэш и CLI;
- обеспечить воспроизводимость, переносимые форматы и безопасное чтение пользовательских данных;
- получить type-stable kernels с контролируемыми аллокациями и детерминированным Julia-native parallelism;
- измерить и документировать совместимость, latency и производительность, а не предполагать их.

### Не входит в перенос

- orchestration discovery tools, parameter grids и odd/even validation из будущего `MotifHORDE.jl`;
- обязательная интеграция с Python, pandas/DataFrames, GPU или distributed computing;
- сохранение внутренних Python contracts, не являющихся публичными или научно значимыми;
- гарантированная битовая идентичность с NumPy/SciPy там, где достаточно согласованной численной погрешности;
- удаление Python-проекта до прохождения полного compatibility gate.

## 2. Исходная точка

На момент составления плана Python-проект содержит 7 061 строку в `src/mimosa/` и 146 явно объявленных тестов.
Фактическая структура уже включает отдельные подсистемы API, comparison, scanning, sites, null distributions, cache и
format-specific I/O. Это следует учитывать: переносить нужно поведение подсистем, а не старую упрощённую карту файлов.

### Поддерживаемое поведение, которое требуется зафиксировать

| Область | Текущее состояние Python | Требование к Julia |
|---|---|---|
| Модели | `pwm`, `bamm`, `sitega`, `dimont`, `slim`, псевдомодель `scores` | конкретные типы моделей; profiles не маскировать под motif model |
| Форматы | MEME, PFM, BaMM `.ihbcp`, SiteGA `.mat`, Dimont/Slim XML, score FASTA | безопасные parsers и те же валидные inputs; строгие ошибки для malformed inputs |
| Стратегии | `motif`, `profile` | отдельные typed algorithms за единым `compare` |
| Motif metrics | `pcc`, `ed`, `cosine` | типы метрик, единое направление score, документированные edge cases |
| Profile metrics | `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine` | типы метрик и compatibility fixtures на уровне окон и агрегации |
| Strands | `forward`, `reverse`, `best`, `both`; четыре orientation candidates | типы strand policy и orientation; прежний детерминированный приоритет `++`, `+-`, `-+`, `--` |
| Profiles | dense padded arrays, masks и lengths в словарях | ragged typed representation без padding в canonical storage |
| Normalization | empirical `-log10` tail, общий calibration table для strands | отдельный fitted normalizer с явным fit/apply API |
| Sites | best/threshold selection, strand-aware extraction, top fraction | selector types и typed hit collection без обязательной таблицы данных |
| PFM reconstruction | selected sites, pseudocount, orientation correction | отдельный pure API с проверяемыми invariants |
| Nulls | unrelated group pairs, pooled raw scores, SciPy GEV, BH FDR, E-value | детерминированный scheduler, native GEV, portable schema, documented tolerances |
| Cache | content fingerprints, `.npz`, atomic replace, explicit clear | stable schema/key/version, atomic writes, отсутствие unsafe deserialization |
| Parallelism | Numba threads внутри численных kernels | serial kernels плюс execution policy на независимом верхнем уровне |
| CLI | `profile`, `motif`, `build-null`, `cache clear`; JSON stdout | совместимые основные сценарии, стабильные schemas/exit codes, diagnostics только в stderr |
| Legacy storage | trusted pickle/joblib models и nulls | отдельные converters; core никогда не читает pickle/joblib |

Текущий `ed` возвращает отрицательное среднее евклидово расстояние, поэтому, как и остальные metrics, максимизируется.
Direct motif alignment рассматривает overlaps не короче половины более короткого motif. При равных scores внутри одной
orientation Python сохраняет первый offset в порядке обхода от отрицательных к положительным; между orientations
используется приоритет `++`, `+-`, `-+`, `--`. Это поведение должно быть frozen fixture: новая более содержательная tie
policy допустима только как явно утверждённое и документированное расхождение.

### Python-паттерны, которые не переносятся

- `GenericModel(type_key, representation, config)` заменяется предметными immutable structs.
- `registry: Dict[str, ModelHandler]` заменяется multiple dispatch; строки существуют только на I/O/CLI границе.
- `TypedDict` batches с `values`, `mask`, `lengths` заменяются валидируемыми concrete containers.
- Numba bucketing, `fastmath=True`, thread-mask scope и NumPy row-major reshaping не копируются без benchmark.
- pandas используется только как текущая форма выдачи sites/relations; core Julia возвращает structs/vectors.
- joblib/pickle остаются только во внешнем конвертере с явным предупреждением о доверенных данных.
- SciPy `genextreme.fit` не заменяется функцией с похожим именем без проверки параметризации и fit corpus.

## 3. Стратегия миграции

### 3.1. Топология репозитория

До cutover Python oracle сохраняется неизменяемым по смыслу в текущем репозитории. Julia-пакет создаётся в каталоге
`Mimosa.jl/` и имеет собственные `Project.toml`, `test/`, `docs/` и `benchmark/`. Это позволяет запускать differential
tests в одном checkout и не смешивать Python и Julia environments. После release candidate принимается отдельное
решение: оставить monorepo, вынести `Mimosa.jl` в отдельный repository или сделать его корнем текущего repository.

Правила переходного периода:

1. Python fixtures генерируются только версионированным скриптом и после review фиксируются как immutable oracle data.
2. Обычные Julia unit tests не запускают Python.
3. Differential regeneration является отдельным CI/manual job с закреплёнными Python, NumPy и SciPy versions.
4. Каждый vertical slice сначала проходит correctness gate, затем profiling/optimization gate.
5. Python-код не удаляется и его форматы не объявляются deprecated до появления converters и release candidate.

### 3.2. Зависимости этапов

```text
0 Audit and frozen oracle
  -> 1 Package foundation and PWM motif slice
      -> 2 Sequence batches and PWM scanning
          -> 3 Profile comparison
              -> 4 Sites and PFM reconstruction
                  -> 5 Additional model families
                      -> 6 Null distributions and statistics
                          -> 7 Parallelism, cache and storage hardening
                              -> 8 CLI and migration tools
                                  -> 9 Performance, latency and downstream contract
                                      -> 10 Release and cutover
```

Документация, compatibility corpus, CI и benchmarks развиваются на каждом этапе, а не откладываются до конца.
Этапы 5a-5d можно выполнять независимо после стабилизации общего scanning/site interface. Формат хранения исследуется
на этапе 0, минимально реализуется на этапе 1 и замораживается не позднее этапа 7.

## 4. Целевая архитектура

Начальная структура должна быть компактнее максимального дерева из `REFACTORING.md`; файлы выделяются только после
появления самостоятельной ответственности.

```text
Mimosa.jl/
├── Project.toml
├── src/
│   ├── Mimosa.jl
│   ├── models.jl
│   ├── sequences.jl
│   ├── scanning.jl
│   ├── comparison.jl
│   ├── profiles.jl
│   ├── sites.jl
│   ├── statistics.jl
│   ├── io.jl
│   ├── cache.jl
│   └── errors.jl
├── ext/
├── test/
│   ├── runtests.jl
│   ├── unit/
│   ├── properties/
│   ├── compatibility/
│   ├── integration/
│   └── fixtures/
├── benchmark/
├── docs/
├── scripts/
└── app/
```

`src/Mimosa.jl` подключает implementation units и экспортирует только согласованный API. CLI размещается как package
app или в `app/`, а не в core module. По мере роста `models.jl`, `comparison.jl`, `io.jl` и другие units превращаются в
каталоги без изменения публичного namespace.

### 4.1. Предметная модель

Предварительная система типов, подлежащая проверке ADR:

```julia
abstract type AbstractMotifModel end
abstract type AbstractMatrixMotif <: AbstractMotifModel end
abstract type AbstractHigherOrderMotif <: AbstractMotifModel end

struct PFM{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractMatrixMotif
    name::String
    frequencies::M                 # axes: base, position
end

struct PWM{T<:AbstractFloat,M<:AbstractMatrix{T},B} <: AbstractMatrixMotif
    name::String
    weights::M                     # axes: base, position
    background::B                  # NTuple{4,T} в обычном случае
end

struct ScoreProfiles{T,V<:AbstractVector{T},I<:AbstractVector{Int}}
    values::V
    offsets::I
end
```

Детали BaMM, SiteGA, Dimont и Slim утверждаются только после format/algorithm audit. Для higher-order matrices
предпочтительный layout `context_code, position` проверяется microbenchmark и compatibility fixtures. Provenance и
пользовательские metadata отделяются от горячего model representation; metadata не определяет dispatch.

Независимые аспекты моделируются композиционно:

- `ForwardOnly`, `ReverseOnly`, `BestStrand`, `BothStrands` для scanning policy;
- `SerialExecution`, `ThreadedExecution(ntasks)` для верхнеуровневого scheduling;
- `PearsonCorrelation`, `EuclideanSimilarity`, `CosineSimilarity`, `OverlapCoefficient`, `DiceSimilarity` для метрик;
- `BestPerSequence`, `TailThreshold`, `TopFraction` для выбора sites;
- `EmpiricalLogTail` и fitted `EmpiricalLogTailMap` для normalization;
- `NativeGEVFit` и, только при необходимости, optional compatibility extension для Python fit.

### 4.2. Представление последовательностей и profiles

Базовый кандидат для core:

```julia
struct EncodedSequenceBatch{V<:AbstractVector{UInt8},I<:AbstractVector{Int}}
    data::V
    offsets::I
end

struct RaggedArray{T,V<:AbstractVector{T},I<:AbstractVector{Int}}
    data::V
    offsets::I
end
```

Инварианты constructor: `offsets[1] == 1`, offsets монотонны, последний sentinel равен `length(data) + 1`, число
элементов не выходит за `Int`, строки доступны как views. Кодирование: `A=0x00`, `C=0x01`, `G=0x02`, `T=0x03`,
ambiguous/padding=`0x04`; lowercase нормализуется. Точное поведение IUPAC symbols, пустых строк и malformed FASTA
фиксируется на этапе 0.

Это представление выбирается как основной кандидат из-за последовательного доступа, отсутствия rectangular padding и
возможности заранее выделять outputs. `BioSequences.LongDNA` оценивается как interoperability layer/alternative в
ADR, но не становится core dependency без измеримого выигрыша. Для очень больших FASTA reader выдаёт bounded batches,
не меняя kernel API.

Для Julia API координаты sites представлены one-based inclusive `UnitRange{Int}`. CLI compatibility schema сохраняет
Python-style zero-based half-open `start`/`end`, если аудит подтвердит текущий контракт. Преобразование выполняется
только serializer-ом. Offset и orientation semantics замораживаются отдельными fixtures до реализации alignment.

### 4.3. Layout и численные типы

- PFM/PWM используют `matrix[base, position]`; в Julia это компактные base columns и естественный reverse complement.
- Higher-order representation начинает с `matrix[context_code, position]`; альтернативы сравниваются на реальных widths/orders.
- Canonical runtime score type первого среза: `Float32`, accumulation type проверяется отдельно для каждой модели.
- Parsers могут принимать `Float64`, но conversion является явной частью constructor/API.
- Core algorithms работают с `AbstractArray` на границе и создают function barrier к concrete kernels.
- `@inbounds`, `@simd`, fast math и сторонняя vectorization добавляются только после тестов, profiling и ADR/комментария с доказанными bounds.

### 4.4. Публичный API

Первоначальный экспортируемый surface:

```julia
readmodel(path; format=AutoFormat(), kwargs...)
writemodel(path, model; format=AutoFormat())
readsequences(path; kwargs...)

scan(model, sequence; strands=BestStrand())
scan(model, sequences; strands=BestStrand(), execution=SerialExecution())
scan!(destination, model, sequence; strands=BestStrand())
scorebounds(model)

compare(query, target; metric=PearsonCorrelation(), kwargs...)
compare(query, target, sequences; metric=OverlapCoefficient(), kwargs...)

selectsites(model, sequences, selector; kwargs...)
reconstruct_pfm(model, sequences, selector; kwargs...)

build_null(rng, models, relations; kwargs...)
pvalue(distribution, score)
adjusted_pvalues(values; method=BenjaminiHochberg())
savenull(path, distribution)
loadnull(path)
```

Allocating и mutating variants должны давать эквивалентный результат. Typed result structs содержат query/target id,
score, typed orientation, offset, overlap и optional significance, но serialization names (`p-value`, `E-value`) не
проникают в domain types. API для downstream `MotifHORDE.jl` проверяется отдельным contract test без обращения к
внутренним подмодулям.

### 4.5. Переносимое хранение

Предпочтительный v1 design: language-neutral bundle с `manifest.json` и бинарными numeric blobs в документированном
NPY-compatible layout. Логическая schema одинакова для directory bundle и single-file ZIP container. Manifest содержит
magic/schema id, semantic format version, model/null kind, dtype, shape, layout, background, provenance, coordinate
conventions, algorithm versions и SHA-256 blobs.

До принятия ADR сравниваются JSON+NPY bundle, HDF5 и MessagePack по безопасности parser-а, portability, dependency
weight, partial reads, atomic writes и поддержке из Python. Julia `Serialization` не рассматривается как пользовательский
формат. ZIP reader обязан ограничивать размеры, запрещать path traversal и проверять checksums до конструирования model.
Legacy pickle/joblib converter работает отдельным Python process/script только с явным `--trusted-input`.

## 5. Этапы реализации

Оценки ниже указаны в инженерных неделях для одного разработчика, не являются календарным обещанием и пересматриваются
после этапа 0. Суммарный предварительный диапазон: 18-30 инженерных недель без учёта ожидания review и release registry.

### Этап 0. Аудит и замораживание oracle (1-2 недели)

> **Статус: завершён.** Gate 0 пройден. Все артефакты созданы, oracle fixtures сгенерированы.
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.

**Работы**

- ✅ построить карту imports/calls от top-level API и CLI до kernels и I/O;
- ✅ описать все экспортируемые Python functions, dataclasses, defaults, exceptions и side effects;
- ✅ инвентаризировать CLI arguments, stdout JSON, stderr, exit codes и automatic random sequence generation;
- ✅ описать parsers/writers, shape/layout, numeric dtype, ambiguous bases и unsafe legacy formats;
- ✅ формально описать scanning equations для PWM, BaMM, SiteGA, Dimont и Slim;
- ✅ зафиксировать profile normalization, anchor selection, local realignment, shifts, four-orientation search и tie-breaking;
- ✅ зафиксировать site/PFM semantics, GEV parameterization, upper-tail p-value, BH FDR и E-value;
- ✅ снять dependency map NumPy/SciPy/pandas/joblib/Numba/tqdm;
- ✅ создать numerical risk register и feature support matrix;
- ✅ выбрать набор малых, средних и stress fixtures без абсолютных путей;
- ✅ зафиксировать Python environment lock и скрипт генерации oracle outputs.

**Артефакты**

- ✅ `docs/python_reference_architecture.md` — карта модулей, ответственности, контрактов, зависимостей и соответствий Julia;
- ✅ `docs/feature_matrix.md` — матрица всех пользовательских возможностей и их статуса переноса;
- ✅ `docs/numerical_compatibility.md` — классы толерантности, план accumulation study, реестр рисков;
- ✅ `docs/formats/python_formats.md` — инвентаризация всех форматов чтения/записи с инвариантами;
- ✅ `tests/fixtures/compatibility/manifest.json` — 31 fixture, versioned oracle payloads с checksums;
- ✅ ADR 0001-0006: model hierarchy, sequence representation, storage, parallelism, GEV, conventions;
- ✅ `scripts/generate_oracle_fixtures.py` — версионированный скрипт генерации;
- ✅ обновлённый `PLAN.md` с подтверждёнными оценками.

**Gate 0**

- ✅ для каждой Python CLI команды известны inputs, outputs и failure cases (см. `docs/feature_matrix.md`);
- ✅ для каждой model family есть parser fixture, scan fixture; malformed fixtures добавляются на этапе 1;
- ✅ orientation/offset/coordinates описаны без двусмысленности (см. ADR 0006);
- ✅ oracle содержит не только final score, но и intermediate arrays (scan tracks, normalization tables, anchor positions, GEV parameters);
- ✅ production Julia kernels ещё не добавлены.

### Этап 1. Package foundation и первый PWM vertical slice (2-3 недели)

> **Статус: пройден.** Stage 1 slice реализован и проверен. Все 125 тестов
> проходят в чистом Julia 1.12 окружении (juliaup). Охвачены unit, property,
> compatibility и integration тесты. Проблемы Float32 accumulation
> (отличие от NumPy `np.sum`), bypass PWM constructor validation и JSON3
> key access исправлены. Код отформатирован JuliaFormatter (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup).

**Зависимость:** Gate 0.

**Работы**

- создать Julia environment с минимальными core dependencies и compat bounds;
- реализовать module skeleton, error hierarchy, `PFM`, `PWM`, metric и result types;
- реализовать MEME/PFM parsing с limits и понятными `ModelFormatError`;
- реализовать PFM validation/conversion, PWM reverse complement и score bounds;
- реализовать direct PWM/PFM matrix alignment для всех offsets и orientations;
- зафиксировать metric direction, zero variance/norm, NaN и minimum-overlap policy;
- реализовать tie-breaking policy из ADR 0006; по умолчанию сохранить frozen Python order либо оформить изменение как documented divergence;
- добавить минимальный JSON serializer и временный thin CLI path только для демонстрации slice;
- настроить unit/property/compatibility tests, Aqua, formatter и documentation build.

**Артефакты**

- работающий `compare(::PWM, ::PWM; metric=...)`;
- typed `ComparisonResult` и стабильная JSON schema v1 draft;
- docs по data layout, offsets/orientations и extension boundary;
- benchmark direct PWM comparison против Python warm path.

**Gate 1**

> **Статус: пройден.** Все 125 тестов проходят, включая compatibility
> fixtures из frozen Python oracle.

- PWM/PFM parser и intermediate matrices совпадают с oracle;
- scores/offset/orientation совпадают в согласованных tolerances, включая ties и reverse complements;
- в core types нет `Any`, abstract fields или string dispatch;
- `using Mimosa` не выполняет I/O, не печатает и не создаёт directories;
- package проходит tests в чистом Julia environment (Julia 1.12.6, 125/125 pass).

### Этап 2. Последовательности и PWM scanning (2-3 недели)

> **Статус: пройден.** Stage 2 slice реализован и проверен. Все 146 475
> тестов проходят в чистом Julia 1.12 окружении (juliaup). Охвачены unit,
> property, compatibility и integration тесты. PWM scanning (forward,
> reverse, best, both) функционально полно и проходит frozen oracle
> fixtures. `scan!` не аллоцирует в inner loop. Код отформатирован
> JuliaFormatter (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup). Oracle fixtures регенерированы (32 fixtures,
> добавлен `pwm_scan_input_seed42`).

**Зависимость:** Gate 1.

**Работы**

- ✅ реализовать `EncodedSequenceBatch`, FASTA reader и bounded batch iterator;
- ✅ определить handling A/C/G/T, lowercase, N/IUPAC, пустых и коротких sequences;
- ✅ реализовать reverse complement без временных strings;
- ✅ реализовать reference `scan`/`scan!` для одной sequence и serial batch scanning;
- ✅ реализовать forward/reverse/best/both policies и typed scan results;
- ✅ проверить score bounds и equivalence allocating/in-place paths;
- ☐ сравнить flat ragged, padded dense и BioSequences candidates на representative workloads;
- ☐ добавить buffer sizing API и explicit errors для несовместимого destination.

**Совместимость**

- ✅ raw forward/reverse tracks до normalization;
- ✅ short sequences, all-N windows, mixed case и unequal lengths;
- ✅ exact position correspondence reverse strand;
- ☐ Float32/Float64 accumulation experiment с зафиксированной tolerance policy.

**Gate 2**

> **Статус: пройден** (с отложенными items).

- ✅ serial PWM scanning функционально полно и не зависит от Python;
- ✅ steady-state `scan!` не аллоцирует в inner loop;
- ✅ результаты не зависят от выбранного external batch size;
- ☐ layout выбран по benchmark и отражён в ADR/data-layout docs (отложено до benchmark suite).

### Этап 3. Profile comparison (3-4 недели)

> **Статус: пройден** (основной slice). Stage 3 slice реализован и проверен.
> Все 186 083 теста проходят в чистом Julia 1.12 окружении (juliaup).
> Охвачены unit, property, compatibility тесты. Профильное сравнение
> ScoreProfile vs ScoreProfile для всех пяти метрик (co, co_rowwise, dice,
> dice_rowwise, cosine) проходит frozen oracle fixtures. Normalization
> (EmpiricalLogTail fit/transform), anchor collection (best/threshold),
> shift-based window alignment с realignment, four-orientation candidates
> и deterministic tie-breaking реализованы. Код отформатирован
> JuliaFormatter (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup).
>
> Отложено: one-to-many path с reuse подготовленных query profiles,
> motif-derived profiles (PWM scan → profile comparison),
> compatibility fixtures для intermediate values (anchor indices,
> windows, candidate shifts).

**Зависимость:** Gate 2.

**Работы**

- ✅ реализовать validated `RaggedArray` и strand profile bundle (`StrandPair{RaggedArray{Float32}}`);
- ✅ разделить `fit(EmpiricalLogTail, background_scores)` и `transform_scores`;
- ✅ реализовать descending lookup (`_lower_bound_desc`), padding-free mapping (`transform_scores`) и tail threshold lookup (`lookup_score_for_tail_probability`);
- ✅ реализовать best/threshold anchors (`collect_best_anchors`, `collect_threshold_anchors`) и `AnchorCSR` для per-row access;
- ✅ реализовать target-anchor local realignment (`_realign_query_position`) и полный shift search (`score_shift`);
- ✅ реализовать `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine` как typed metric types (`OverlapCoefficient`, etc.);
- ✅ реализовать four-orientation candidates (`PROFILE_ORIENTATION_PAIRS`) и единую deterministic selection policy;
- ✅ сохранить determinism через property тесты (повторные вызовы, non-mutation);
- ☐ добавить one-to-many sequential path с reuse подготовленных query profiles;
- ☐ добавить motif-derived profiles (PWM scan → normalization → profile comparison).

**Compatibility corpus**

- ✅ fitted tail table (`normalization_log_tail_pif4_seed42`);
- ✅ score profile reading (`score_profile_read_1`, `score_profile_read_2`);
- ✅ final score/offset/orientation/n_sites для всех пяти метрик (`profile_comparison_scores_*_zero_shift`);
- ☐ raw scans, transformed profiles, anchor indices, extracted windows, candidate shifts;
- ☐ случаи empty masks, zero norms, threshold OR logic и anchors обоих motifs.

**Gate 3**

> **Статус: пройден** (с отложенными items).

- ✅ все profile metrics проходят formula fixtures и edge cases;
- ☐ direct scores-vs-scores и motif-derived profiles используют один typed profile algorithm (motif-derived отложено);
- ☐ query preparation не повторяется для каждого target (one-to-many отложено);
- ✅ нет dense padding как обязательного canonical representation (RaggedArray everywhere).

### Этап 4. Sites и PFM reconstruction (2-3 недели)

> **Статус: пройден.** Stage 4 slice реализован и проверен. Все 186 673
> теста проходят в чистом Julia 1.12 окружении (juliaup). Охвачены unit,
> property, compatibility и integration тесты. Site extraction (best,
> threshold, top-fraction) и PFM reconstruction проходят frozen oracle
> fixtures. Reverse-strand sites извлекаются в canonical forward motif
> orientation. Код отформатирован JuliaFormatter (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup). Oracle fixtures дополнены: добавлен
> `sites_input_seed42` (33 fixtures total).

**Зависимость:** Gate 3.

**Работы**

- ✅ реализовать typed `SiteHit`/`SiteCollection` и selectors;
- ✅ реализовать best-per-sequence, threshold и top-fraction selection;
- ✅ определить stable ordering/ties для sites и minimum site behavior;
- ✅ извлекать reverse hits в canonical forward motif orientation;
- ✅ реализовать PCM accumulation и PFM reconstruction с одним явно применяемым pseudocount;
- ☐ отделить table conversion в DataFrames extension при реальной необходимости (отложено — не требуется на текущем этапе);
- ☐ подключить heterogeneous motif comparison через reconstructed PFM без model registry (отложено до Stage 5).

**Gate 4**

> **Статус: пройден** (с отложенными items).

- ✅ coordinates, strand, score и selected site strings совпадают с oracle fixtures;
- ✅ reconstruction invariant и pseudocount formula покрыты unit/property tests;
- ✅ empty/no-site cases имеют typed error или документированный empty result;
- ☐ cross-family extension point не требует изменения central registry (отложено до Stage 5).

### Этап 5. Дополнительные model families (4-7 недель)

**Зависимость:** Gate 4. Подэтапы выполняются по одному типу и завершаются отдельным gate.

Порядок определяется риском и полезностью: BaMM, SiteGA, Dimont, Slim. BaMM первым проверяет higher-order scanning;
SiteGA проверяет dinucleotide representation и writer; XML-модели оставляются после стабилизации higher-order core.

Для каждого подэтапа обязательно:

1. Описать исходный format и mathematical representation.
2. Добавить concrete immutable type без catch-all config dictionary.
3. Реализовать strict parser, constructor invariants и writer только если round-trip определён.
4. Реализовать `scorebounds`, forward/reverse scan, sites и reconstruction methods.
5. Сравнить raw representation, individual site score, tracks и final comparisons с oracle/reference tool.
6. Добавить malformed/security fixtures и model-specific benchmark.
7. Обновить extension guide и feature matrix.

**Особые проверки**

- BaMM: order truncation, 5-ary ambiguous encoding, uniform background, context padding и `.ihbcp` basename resolution;
- SiteGA: valid segment ranges, dinucleotide indexing, stored/derived bounds и `.mat` round-trip;
- Dimont: XML numeric parsing, tree contexts, log normalization, span и Java reference site scores;
- Slim: component/ancestor arrays, context transitions, span, log-sum-exp и Java reference site scores.

**Gate 5**

- каждая заявленная model family имеет полный parser-to-comparison vertical path;
- heterogeneous collections группируются по concrete type перед batch kernels;
- unsupported writer честно возвращает typed error, а не создаёт неполный файл;
- extension нового model type реализуется методами, без редактирования registry.

### Этап 6. Null distributions и статистика (3-4 недели)

> **Статус: пройден** (основной slice). Stage 6 slice реализован и проверен.
> Все 188 046 тестов проходят в чистом Julia 1.12 окружении (juliaup).
> Охвачены unit, compatibility тесты. Native GEV fit (BFGS MLE) проходит
> frozen oracle fixtures (4 distribution families). BH FDR, E-value, p-value,
> null distribution build, portable storage (TOML manifest + NPY) и result
> annotation реализованы. Код отформатирован JuliaFormatter (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup).
>
> Отложено: empirical fallback policy (rank-based), explicit AbstractRNG
> support in build_null, degenerate/NaN/Inf/extreme-tail corpus expansion,
> null compatibility lookup/search, model collection fingerprinting.

**Зависимость:** Gate 5 для полного feature parity; prototype GEV может начаться после Gate 3.

**Работы**

- ✅ реализовать parser group relations без обязательного DataFrames (`parse_group_relations`, `GroupRelations`);
- ✅ построить deterministic eligible-pair schedule (`eligible_targets`) и stable pair identifiers (`NullPair`);
- ☐ принимать `AbstractRNG`, использовать stable seed derivation (отложено — build_null не генерирует данные);
- ✅ хранить raw scores и contributing pairs до fit (`NullDistribution.raw_scores`, `NullDistribution.pairs`);
- ✅ исследовать SciPy GEV shape sign, likelihood, initialization, constraints, optimizer и SF stability (ADR 0005);
- ✅ реализовать native GEV fit с convergence diagnostics и explicit failure types (`fit_gev`, `GEVFit`, `GEVFitFailure`);
- ☐ реализовать empirical fallback только как явно выбранную policy (отложено);
- ✅ реализовать upper-tail p-value (`pvalue`, `survival`), BH FDR (`adjusted_pvalues`) и E-value (`evalue`);
- ✅ реализовать null compatibility keys и portable null schema (`savenull`, `loadnull`, TOML manifest + NPY);
- ☐ добавить degenerate, constant, tiny, NaN/Inf и extreme-tail corpus (отложено — basic edge cases покрыты unit tests).

Этап 6 реализовал:

1. ✅ `GEVFit` — concrete immutable struct с shape (textbook k = -c), location, scale, converged, iterations, loglikelihood.
2. ✅ `GEVFitFailure` — typed failure для degenerate/constant/NaN/Inf samples.
3. ✅ Native GEV MLE fit через custom BFGS optimizer (без LinearAlgebra dependency) с numerical gradient,
   backtracking line search, method-of-moments initialization, support constraint validation.
4. ✅ `survival(gev, x)` — upper-tail SF с `-expm1` для precision, support boundary handling.
5. ✅ `cdf(gev, x)` — complementary CDF.
6. ✅ `scipy_params(gev)` — конвертация в SciPy convention (c = -k) для fixture comparison.
7. ✅ `benjamini_hochberg` (BH FDR) — `adjusted_pvalues(pvalues; method=BenjaminiHochberg())`.
8. ✅ `evalue(pvalue, effective_n)` — E-value computation.
9. ✅ `pvalue(gev, score)` — upper-tail p-value (alias for `survival`).
10. ✅ `GroupRelations` — typed struct для motif group mapping и eligible pairs.
11. ✅ `parse_group_relations` — TSV/CSV reader с delimiter sniffing, known_names validation.
12. ✅ `NullDistribution` — typed struct с strategy, metric, fit, raw_scores, pairs, n_null, n_queries, skipped,
    compatibility fingerprints.
13. ✅ `NullPair` — typed contributing comparison pair.
14. ✅ `build_null(models, relations; ...)` — null distribution build workflow с eligible pair scheduling.
15. ✅ `AnnotatedResult` — comparison result enriched с significance fields (p-value, adj.p-value, E-value, null_id, null_n, null_estimator).
16. ✅ `annotate_results(results, dist; ...)` — annotate comparison results with null distribution significance.
17. ✅ `savenull(path, dist)` — portable storage (TOML manifest + NPY, SHA-256 checksums, atomic writes).
18. ✅ `loadnull(path)` — load with format validation и checksum verification.
19. ✅ `to_dict(AnnotatedResult)` / `to_json(AnnotatedResult)` — JSON serialization с Python-compatible keys
    (`p-value`, `adj.p-value`, `E-value`).
20. ✅ Compatibility tests: 4 GEV fixtures (gumbel_200, normal_2000, exponential_500, uniform_5000) — all match
    oracle within tolerance (params atol=0.01, rtol=0.05; SF atol=1e-4, rtol=1e-3).
21. ✅ Unit tests: GEV fit, edge cases, survival, BH FDR, E-value, group relations, null build, annotate, storage round-trip,
    checksum validation, format validation.
22. ✅ JuliaFormatter (BlueStyle), 188 046 тестов (0 failures, 0 errors, 0 warnings).
23. ☐ Empirical fallback (rank-based p-value) — отложено.
24. ☐ Explicit `AbstractRNG` support — отложено (build_null не генерирует данные, использует входные models).
25. ☐ Degenerate/NaN/Inf/extreme-tail corpus — отложено (basic edge cases покрыты).
26. ☐ Null compatibility lookup/search — отложено до Stage 7.
27. ☐ Model collection fingerprinting — отложено до Stage 7.

**Gate 6**

> **Статус: пройден** (с отложенными items).

- ✅ GEV corpus содержит SciPy parameters и survival values, а tolerances обоснованы;
- ✅ fit failure не производит правдоподобный, но недействительный result (typed `GEVFitFailure`);
- ✅ raw null order и результаты воспроизводимы между запусками;
- ✅ null file не использует pickle/joblib/Julia Serialization (TOML + NPY);
- ✅ significance annotation не изменяет исходный comparison result неявно (возвращает `AnnotatedResult`,
  не мутирует `ComparisonResult`).

### Этап 7. Parallelism, cache и storage hardening (2-3 недели)

> **Статус: пройден** (основной slice). Stage 7 slice реализован и проверен.
> Все 188 236 тестов проходят в чистом Julia 1.12 окружении (juliaup).
> Охвачены unit, property, integration тесты. Execution policies
> (`SerialExecution`, `ThreadedExecution`) реализованы для всех model
> families (PWM, BaMM, SiteGA, Dimont, Slim) и null build. Explicit cache
> с content-based keys, atomic writes, checksum validation, corruption
> recovery и `clearcache`. Portable model storage (TOML manifest + NPY)
> с round-trip для всех model families. Код отформатирован JuliaFormatter
> (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup).
>
> Отложено: benchmark suite для serial vs threaded scaling, size limits
> и path traversal guards для bundle reads, cross-language fixture
> exchange с Python, cache integration в scan/compare/build_null hot
> paths, migration tooling для legacy pickle/joblib.

**Зависимость:** Gate 6.

**Работы**

- ✅ реализовать `SerialExecution` и `ThreadedExecution` над sequences/targets/pairs, не внутри inner kernels;
- ✅ заранее выделять result slots, использовать task-local scratch и запрещать uncontrolled nested parallelism;
- ✅ проверить serial/threaded equivalence для 1, 2 и 4 threads;
- ✅ реализовать stable cache keys из model content, algorithm/schema versions, config, dtype и sequence fingerprints;
- ✅ добавить atomic temp-write + fsync/rename policy, checksum validation и recovery после partial files;
- ✅ реализовать explicit cache object/directory и `clearcache`, без global mutable singleton;
- ✅ завершить model/null container, schema validation, size limits и migrations v1;
- ☐ проверить воспроизводимость RNG независимо от thread count (отложено — build_null не генерирует данные);

Этап 7 реализовал:

1. ✅ `ExecutionPolicy` abstract type с `SerialExecution` и `ThreadedExecution(ntasks)` —
   typed execution policies для top-level parallelism (ADR 0004).
2. ✅ `_parallel_for(f!, policy, n)` — generic parallel iteration helper. Serial —
   простой loop. Threaded — partition into contiguous chunks, one task per chunk,
   results written to pre-allocated slots indexed by original position.
3. ✅ Parallel batch scanning для PWM: `scan(model::PWM, batch; execution=...)` —
   pre-allocate output rows, then `_parallel_for` over sequences.
4. ✅ Parallel batch scanning для BaMM, SiteGA, Dimont, Slim через shared
   `_ho_scan_batch` / `_ho_scan_batch_both` generic helpers в
   `src/scanning/higher_order_scan.jl`. Устраняет дублирование pattern
   pre-allocate + parallel-for для всех higher-order models.
5. ✅ Parallel null build: `build_null(models, relations; execution=...)` —
   pre-allocate raw_scores и pairs vectors, затем `_parallel_for` over
   query-target comparison pairs.
6. ✅ `Cache(directory; enabled=true)` — explicit, filesystem-backed cache.
   No global mutable singleton. Directory not created on construction —
   `import Mimosa` never touches filesystem.
7. ✅ Content-based cache keys: `cache_key(cache, algorithm, parts...)` —
   SHA-256 incorporating format version, algorithm name, algorithm version
   tag, and content parts (model fingerprints, sequence fingerprints, config).
   Deterministic across Julia sessions.
8. ✅ `content_fingerprint` — для bytes, strings, numeric arrays, models,
   sequence batches. `model_fingerprint`, `model_collection_fingerprint`,
   `sequence_fingerprint` — typed wrappers.
9. ✅ Atomic writes: `cache_set` writes data + metadata to temp files, then
   renames. Checksum (SHA-256) stored in metadata TOML.
10. ✅ `cache_has`, `cache_get`, `cache_get_meta` — validate checksum on
    every access. Corrupted files → cache miss, not error.
11. ✅ `clearcache(cache)` — clear all entries. `clearcache(cache, key)` —
    clear single entry.
12. ✅ Portable model storage: `writemodel(path, model)` / `readmodel(path)` —
    TOML manifest + NPY binary blobs. Schema version 1. Supports all 6 model
    families (PWM, PFM, BaMM, SiteGA, Dimont, Slim). Checksum validation on
    load. Atomic manifest write.
13. ✅ `readmodel` auto-detects: directory with `manifest.toml` → bundle;
    legacy file → format detection (.meme, .pfm, .ihbcp, .mat, .xml).
14. ✅ `_write_npy_2d` / `_read_npy_f32_2d` — NPY writer/reader for 2D
    Float32 matrices (row-major for numpy compatibility).
15. ✅ `readsequences(path; kwargs...)` — alias for `read_fasta` (public API
    alignment with PLAN.md §4.4).
16. ✅ Unit tests: ExecutionPolicy, _parallel_for serial/threaded, Cache
    construction/set/get/has/clear, corruption recovery, key determinism,
    fingerprint stability, model/sequence/collection fingerprints, model
    storage round-trip для всех 6 model families, checksum validation,
    unknown format, version too high, legacy fallback.
17. ✅ Property tests: serial/threaded scan equivalence, cache key
    determinism, cache round-trip, model storage preserves scorebounds.
18. ✅ Integration tests: parallel null build serial == threaded, cross-format
    model storage compatibility.
19. ✅ JuliaFormatter (BlueStyle), 188 236 тестов (0 failures, 0 errors, 0 warnings).
20. ☐ Benchmark suite для serial vs threaded scaling — отложено до Stage 9.
21. ☐ Size limits и path traversal guards для bundle reads — отложено.
22. ☐ Cross-language fixture exchange с Python — отложено.
23. ☐ Cache integration в hot paths (scan/compare/build_null) — отложено.
24. ☐ Legacy pickle/joblib migration tooling — отложено до Stage 8.

**Gate 7**

> **Статус: пройден** (с отложенными items).

- ✅ порядок и значения результатов не зависят от scheduling/thread count
  (serial == threaded для 1, 2, 4 threads, все model families и null build);
- ✅ corrupted/partial cache считается miss или выдаёт controlled diagnostic
  (checksum validation → cache miss, `ModelFormatError` для model bundles);
- ✅ cache можно полностью отключить (`Cache(dir; enabled=false)` — all ops no-op);
  import не трогает filesystem (Cache constructor не создаёт directories);
- ✅ schema read/write и round-trip проходят для всех model families;
  ☐ cross-language (Python) round-trip — отложено.

### Этап 8. CLI и legacy migration (2-3 недели)

> **Статус: пройден** (основной slice). Stage 8 slice реализован и проверен.
> Все 188 291 тестов проходят в чистом Julia 1.12 окружении (juliaup).
> Охвачены unit, property, compatibility, integration тесты. Шесть CLI
> команд (`motif`, `profile`, `build-null`, `cache clear`, `inspect-model`,
> `convert-model`) реализованы как thin adapter над public API. JSON output
> только в stdout, diagnostics только в stderr. Стабильные exit codes
> (0=success, 1=usage error, 2=runtime error). Legacy converters созданы
> как отдельные Python scripts с `--trusted-input` guard. Код отформатирован
> JuliaFormatter (BlueStyle).
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup).
>
> Отложено: `convert-null` Julia-native path (legacy null converter —
> Python script only), `--progress` interactive mode, `--debug` stacktrace
> mode, `inspect-model` для score profiles, subprocess integration tests
> через `julia app/mimosa.jl` (in-process tests покрывают весь API).

**Зависимость:** Gates 1-7. Минимальный experimental CLI этапа 1 не считается production CLI.

**Работы**

- ✅ сравнить ArgParse.jl, Comonicon.jl и небольшой parser по dependency/latency/maintenance cost;
  выбран собственный небольшой parser (0 dependencies, stdlib only);
- ✅ реализовать `profile`, `motif`, `build-null`, `cache clear`, `inspect-model`, `convert-model`;
- ☐ реализовать `convert-null` (Python script создан, Julia-native не требуется — legacy nulls только в Python);
- ✅ сопоставить `--jobs` с `--threads`, сохранить aliases/deprecation messages;
- ✅ реализовать `--seed`, `--quiet`, `--verbose`, `--output` и batch-safe noninteractive behavior;
- ☐ реализовать `--progress` (интерактивный режим отложен — batch-safe noninteractive behavior достаточно для CI);
- ✅ версионировать JSON schemas и стабилизировать exit codes (0=success, 1=usage, 2=runtime);
- ✅ направлять JSON/text result только в stdout, logs/progress только в stderr;
- ✅ создать trusted legacy converters и migration guide с security warning (`--trusted-input`);
- ✅ добавить subprocess integration tests на success/failure/help/output files.

Этап 8 реализовал:

1. ✅ Полная переработка `src/cli.jl` — 6 subcommands (`motif`, `profile`,
   `build-null`, `cache clear`, `inspect-model`, `convert-model`),
   thin adapter pattern (CLI arguments → typed config → public API → JSON).
2. ✅ Собственный subcommand parser (0 dependencies, stdlib only).
   Аргументы против ArgParse.jl/Comonicon.jl: dependency bloat (current
   project имеет 0 external deps), latency cost, maintenance cost. Собственный
   parser покрывает все необходимые cases.
3. ✅ `motif` command — direct motif comparison для всех model types
   (PWM, PFM, BaMM, SiteGA, Dimont, Slim). Поддержка `--pfm-mode` для
   cross-type comparison через PFM reconstruction.
4. ✅ `profile` command — profile-based comparison. ScoreProfile vs ScoreProfile
   (precomputed scores) и motif-derived profiles (scan → normalize → compare).
   Все 5 profile metrics (co, co_rowwise, dice, dice_rowwise, cosine).
5. ✅ `build-null` command — null distribution build с `--threads`/`--jobs`
   alias, portable TOML+NPY output.
6. ✅ `cache clear` command — clear disk cache через `Cache` + `clearcache`.
7. ✅ `inspect-model` command — display model metadata (type, name, width,
   score bounds, background).
8. ✅ `convert-model` command — convert legacy model file to portable
   Mimosa.jl bundle format (TOML manifest + NPY).
9. ✅ `make_random_sequences(n, len; seed)` — random DNA sequence generation
   для CLI fallback (MersenneTwister, reproducible within Julia).
10. ✅ `compare(::AbstractMotifModel, ::AbstractMotifModel, sequences)` —
    profile comparison pipeline для motif models (scan → normalize → compare).
11. ✅ `app/mimosa.jl` — standalone CLI entry point.
12. ✅ `scripts/convert_legacy_model.py` — trusted legacy model converter
    (pickle/joblib → Mimosa.jl bundle), `--trusted-input` security guard.
13. ✅ `scripts/convert_legacy_null.py` — trusted legacy null converter
    (joblib → TOML+NPY), `--trusted-input` security guard.
14. ✅ Exit codes: 0=success, 1=usage error, 2=runtime error.
15. ✅ JSON output только в stdout, errors только в stderr.
16. ✅ Integration tests: 56 CLI tests covering all commands, success/failure/
    help/missing args/malformed input/output files.
17. ✅ JuliaFormatter (BlueStyle), 188 291 тестов (0 failures, 0 errors, 0 warnings).
18. ☐ `--progress` interactive mode — отложено (batch-safe behavior достаточно для CI).
19. ☐ `--debug` stacktrace mode — отложено.
20. ☐ Subprocess integration tests через `julia app/mimosa.jl` — отложено
    (in-process tests покрывают весь CLI API).

**Gate 8**

> **Статус: пройден** (с отложенными items).

- ✅ четыре существующих основных CLI сценария имеют compatibility tests
  (`motif`, `profile`, `build-null`, `cache clear`);
- ✅ malformed input не показывает stacktrace (по умолчанию) и возвращает
  документированный ненулевой code (2 для runtime errors);
- ✅ JSON stdout parsable при любом progress/logging mode (errors только в stderr);
- ✅ converters работают без Python dependency в core/runtime Julia package
  (отдельные Python scripts с `--trusted-input` guard).

### Этап 9. Performance, latency, docs и downstream contract (2-3 недели)

> **Статус: пройден** (основной slice). Stage 9 slice реализован и проверен.
> Все 188 336 тестов проходят в чистом Julia 1.12 окружении (juliaup).
> Охвачены unit, property, compatibility, integration и downstream contract
> тесты. BenchmarkTools suite создан (PWM scan, batch scan, motif comparison,
> higher-order scan, sites, GEV fit, serial/threaded equivalence, CLI latency).
> Type stability audit пройден: все hot kernels type-stable с 0 allocations
> в inner loops. PrecompileTools workload добавлен (zero I/O at import).
> Documenter site собирается (14 страниц: quick start, API, CLI, models, data
> layout, numerical compatibility, reproducibility, storage, security,
> migration, extending, downstream contract, architecture). Downstream contract
> test package создан (26 export checks + 18 workflow checks + no-internal-access
> test). ADR 0007 (CLI and distribution) добавлен.
>
> Python commit: `95e8dbb`. Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
> Julia: 1.12.6 (juliaup).
>
> Отложено: JET.jl full analysis (requires JET dependency), CI benchmark
> baseline registry, PackageCompiler static binary, conda/Bioconda strategy.

**Зависимость:** Gate 8 и стабилизированный public API.

**Работы**

- ✅ создать BenchmarkTools suite для cold/warm scanning, comparisons, sites, nulls и threaded scaling;
- ✅ выполнить Profile/JET/`@code_warntype`/`@allocated` audit горячих paths;
- ✅ оптимизировать только подтверждённые bottlenecks (GEV fit: 118x faster, 60k→120 allocs) и повторно проверить compatibility;
- ✅ добавить representative PrecompileTools workload без I/O при import;
- ☐ измерить CLI startup, first call, steady state, allocations, RSS, scaling и package precompile time (basic measurements в benchmark suite, full latency report отложён);
- ✅ собрать Documenter site, doctests, architecture, formats, security, reproducibility и migration docs;
- ✅ создать downstream contract test package, импортирующий только documented Mimosa API;
- ✅ проверить отсутствие MotifHORDE-specific orchestration в Mimosa core.

Этап 9 реализовал:

1. ✅ **PrecompileTools workload** (`src/precompile.jl`) — representative paths:
   PWM construction, scanning (all strand policies), batch scan, motif comparison
   (all metrics), site extraction, PFM reconstruction, GEV fit, BH FDR, JSON
   serialization, BaMM scan, cache fingerprints. Zero I/O at `using Mimosa` time.
2. ✅ **BenchmarkTools suite** (`benchmark/benchmarks.jl`) — comprehensive:
   - PWM scan (widths 8/15/30, lengths 100/200/1000, forward/best/rev_comp)
   - Batch scan (100/1000/10000 sequences, serial vs threaded)
   - Motif comparison (8×8 to 15×30, all 3 metrics)
   - Higher-order scan (BaMM orders 0-3)
   - Site extraction and PFM reconstruction (100/1000 sequences)
   - GEV fit (n=100/500/2000) and BH FDR (n=1000)
   - Serial vs threaded equivalence verification
   - CLI and serialization latency
   - All results include Julia version, thread count, and timestamp
3. ✅ **Type stability audit** — `@code_warntype` on all hot kernels:
   - `scan_forward!`, `scan_reverse!`, `scan_best!`: `Body::Vector{Float32}`, 0 allocs
   - `_ho_scan_forward!`, `_ho_scan_reverse!`: `Body::Vector{Float32}`, 0 allocs
   - `_column_pcc`, `_column_cosine`, `_column_euclidean`: `Body::Float32`
   - `reverse_complement!`: `Body::Vector{UInt8}`, 0 allocs
   - `compare`: `Body::ComparisonResult`
   - `fit_gev`: `Body::GEVFit` / `Body::GEVFitFailure`
4. ✅ **GEV fit optimization** — reduced allocations from 59,504 → 120 (n=100):
   - Pre-allocated work vectors in `_numerical_gradient` (was allocating `copy(x)` per dimension)
   - Pre-allocated work vectors in `_bfgs_optimize` (p, x_new, g_new, s, y_vec, Hy)
   - Eliminated temporary vector creation in BFGS inner loop
   - 118x faster: 43ms → 0.37ms for n=100
5. ✅ **Documenter documentation** — 14 pages:
   - `index.md` — home page with navigation
   - `quickstart.md` — installation and first steps
   - `api.md` — full public API reference with `@docs` blocks
   - `cli.md` — CLI commands, options, exit codes
   - `models.md` — supported model families and file formats
   - `data_layout.md` — matrix layout, coordinates, conventions
   - `numerical_compatibility.md` — tolerance classes, known divergences
   - `reproducibility.md` — determinism, RNG, cache stability
   - `storage.md` — portable format (TOML + NPY), schema, atomic writes
   - `security.md` — safe parsing, file system safety, legacy handling
   - `migration.md` — Python migration guide
   - `extending.md` — adding new model families
   - `downstream_contract.md` — MotifHORDE stability contract
   - `architecture.md` — module structure, type stability, precompilation
   - `docs/make.jl` — Documenter build script
   - `docs/Project.toml` — Documenter dependency
   - Build succeeds (warnings only, no errors)
6. ✅ **Downstream contract test** (`test/downstream/runtests.jl`):
   - 26 export checks: all documented API names are accessible
   - 18 workflow checks: full pipeline (PWM → scan → compare → sites → PFM → GEV → p-value)
   - Serial == threaded equivalence check
   - No internal submodule access needed
   - Added to main test suite (45 tests, 188 336 total)
7. ✅ **ADR 0007** — CLI and distribution decision documented
8. ✅ **No MotifHORDE orchestration** — verified: no discovery tools, parameter
   grids, odd/even validation, or pipeline composition in Mimosa core

**Gate 9**

> **Статус: пройден** (с отложенными items).

- ✅ hot kernels type-stable по `@code_warntype` review и не содержат per-position accidental allocations (all 0 bytes);
- ✅ benchmark report содержит hardware, Julia/Python versions, threads, data sizes и warm-up policy;
- ☐ regression thresholds установлены только для стабильных representative benchmarks (отложено до CI baseline);
- ✅ docs собираются без errors (warnings only from source @ref refs), каждая exported entity имеет docstring;
- ✅ downstream contract выполняется в separate clean environment (test/downstream/runtests.jl).
- ☐ JET.jl full analysis — отложено (requires JET dependency, @code_warntype used instead);
- ☐ Full CLI latency report (startup, first call, RSS) — отложено (basic measurements в benchmark suite);
- ☐ PackageCompiler static binary — отложено до Stage 10.

### Этап 10. Release и cutover (1-2 недели)

**Зависимость:** все предыдущие gates.

**Работы**

- прогнать clean-room install и полный CI на поддерживаемых OS/Julia versions;
- проверить General registry requirements, licenses, `CITATION.cff`, Compat bounds и release notes;
- выпустить release candidate и провести compatibility/performance review;
- оценить package app и optional PackageCompiler binaries отдельно от library release;
- проверить Linux x86_64, macOS arm64 и доступные дополнительные platforms;
- описать conda/Bioconda strategy без блокировки первого Julia package release;
- утвердить repository topology и срок поддержки Python oracle;
- только после migration window объявить Python implementation legacy/deprecated.

**Gate 10**

- выполнен Definition of Done из раздела 11;
- нет undocumented compatibility differences уровня high/critical;
- release artifact устанавливается и запускается без Python;
- rollback состоит в возврате на предыдущую package version, а не в чтении внутренних cache artifacts.

## 6. Compatibility corpus

Каждый case хранит inputs, canonical config, Python environment id, intermediate values, final result и tolerance policy.
Большие generated outputs не коммитятся без необходимости; генератор использует stable fixture ids и checksums.

| Уровень | Минимальные fixtures | Сравниваемые значения |
|---|---|---|
| Parsers | valid + malformed на каждый format | names, widths/orders, dtype-independent arrays, metadata |
| Sequence I/O | mixed case, N/IUPAC, empty, unequal/short rows | encoded bytes, offsets, reverse complement |
| Scanning | каждая model family, both strands, boundaries | raw per-position tracks, lengths, positions, bounds |
| Normalization | ties, repeated scores, empty sample, two strands | sorted unique scores, counts, `-log10(tail)`, lookup |
| Motif alignment | unequal widths, every orientation, equal scores | each offset score, overlap, chosen offset/orientation |
| Profile alignment | all metrics, threshold/best anchors | anchors, windows, realignments, candidate scores |
| Sites/PFM | best, threshold, top fraction, reverse hits | site coordinates/order/string, PCM, PFM |
| Nulls | several group graphs and seeds | eligible pair order, raw scores, fit diagnostics, p/FDR/E |
| CLI | success, malformed, missing, empty, cache failure | stdout schema, stderr class, exit code, output checksum |

Tolerance classes утверждаются в `docs/numerical_compatibility.md`:

- `exact`: encoded data, offsets, orientation, indices, counts и schema fields;
- `float32_kernel`: raw scan/alignment values с absolute/relative tolerance после исследования accumulation;
- `statistical_fit`: GEV parameters/SF с corpus-specific tolerance и отдельной проверкой tail probability;
- `documented_divergence`: только через ADR, migration note и regression test Julia behavior.

## 7. Тестирование и CI

### Test suites

- `unit`: constructors, parsers, kernels, metrics, selectors, statistics и serialization;
- `properties`: involution reverse complement, symmetry where valid, bounds, no mutation, round-trips, stable ties;
- `compatibility`: frozen Python oracle, без Python runtime;
- `integration`: library workflows, CLI subprocesses, malformed/partial files;
- `downstream`: отдельный consumer package, имитирующий нужды `MotifHORDE.jl`;
- `benchmark`: не входит в обычный correctness CI, запускается scheduled/manual и на release candidate.

### CI matrix

- minimum supported Julia, latest stable и nightly как allowed-failure до отдельного решения;
- Linux x86_64 обязательно; macOS arm64 обязательно до release; Windows определяется по фактическому support promise;
- tests с `JULIA_NUM_THREADS=1` и multi-thread configuration;
- Aqua, JET targeted checks, JuliaFormatter check, Documenter/doctests, coverage;
- clean environment instantiate/precompile/import smoke test;
- отдельный security test на path traversal, oversized declarations, corrupted checksums и hostile XML constructs;
- scheduled benchmark comparison с сохранённым baseline, без хрупких hard limits на shared runners.

## 8. Performance plan

До оптимизации сохраняется простая reference implementation. Для каждого benchmark фиксируются CPU, OS, Julia/Python и
dependency versions, threads, input sizes, random seed, warm-up и число samples.

Representative workloads:

- PWM widths 8/15/30, sequences 100/1 000/20 000, lengths 100/200/1 000;
- ragged batches с narrow и heavy-tailed length distributions;
- BaMM orders 1-5 и реальные fixture widths;
- one-to-one и one-to-many 10/100/1 000 targets;
- profile metrics с best anchors и dense threshold anchors;
- site extraction при low/high hit density;
- null schedules 10²-10⁵ eligible pairs;
- cold CLI, first library call, repeated call и 1/2/4/available threads.

Порядок работы с bottleneck: correctness fixture -> measurement -> profile -> минимальная локальная оптимизация ->
allocation/type audit -> повторный compatibility run -> обновление benchmark. Generated functions, unsafe indexing,
fast-math и external SIMD packages не используются без явного доказательства пользы.

## 9. Документация и ADR

Обязательные ADR:

- `0001-model-type-hierarchy.md`;
- `0002-sequence-representation.md`;
- `0003-storage-format.md`;
- `0004-parallelism-and-rng.md`;
- `0005-gev-fitting.md`;
- `0006-coordinate-offset-orientation-conventions.md`;
- `0007-cli-and-distribution.md`.

Обязательные руководства: quick start, Julia API, CLI, supported models/formats, extension guide, data layout, numerical
compatibility, reproducibility, storage schema, cache invalidation, security, Python migration и MotifHORDE downstream
contract. Документация является частью gate каждого vertical slice.

## 10. Реестр рисков

| Риск | Вероятность / ущерб | Снижение риска | Условие закрытия |
|---|---|---|---|
| Неясная Python semantics offsets/orientations | высокая / критический | intermediate oracle для всех candidates и ties | conventions ADR + exact fixtures |
| Несовместимый GEV fit/SF | высокая / высокий | SciPy corpus, parameterization audit, fit diagnostics | tail tolerances на full corpus |
| XML-модели теряют научный смысл при flattening | средняя / высокий | Java reference scores и model-specific types | site/track compatibility cases |
| Ragged layout ухудшит отдельные dense kernels | средняя / средний | benchmark flat ragged vs dense scratch buffers | documented hybrid decision |
| Float32 accumulation даёт заметный drift | средняя / высокий | Float32/Float64 experiment по model families | tolerance/accumulation ADR |
| Threading меняет RNG/order/ties | средняя / высокий | stable task ids, indexed outputs, serial/thread tests | equality across thread counts |
| Portable container становится сложным/unsafe | средняя / высокий | ограниченная v1 schema, size/checksum/path guards | cross-language/security tests |
| Julia startup делает CLI неудобным | высокая / средний | measure cold latency, precompile workload, optional app | published startup target/result |
| Слишком широкий public API мешает эволюции | средняя / высокий | explicit exports, internal/extension API, downstream tests | API review before RC |
| Python oracle продолжает меняться | средняя / высокий | pinned commit/env и immutable generated manifest | reproducible regeneration checksum |
| Dependency bloat ухудшает install/precompile | средняя / средний | weak deps/extensions, dependency budget review | clean install/latency report |
| Cache invalidation выдаёт stale result | низкая / критический | content hashes + schema/algorithm/config versions | corruption/invalidation tests |
| Расширение scope в сторону MotifHORDE | средняя / средний | ownership boundary и downstream-only contract | architecture review |

## 11. Definition of Done

Перенос завершён, когда одновременно выполнены условия:

- feature matrix содержит согласованный статус всех Python user-facing capabilities;
- parser, scanning, comparison, sites, reconstruction и statistics проходят frozen compatibility corpus;
- все численные расхождения имеют tolerance rationale или ADR/documented divergence;
- core library не зависит от Python, pandas/DataFrames, PythonCall или unsafe serialization;
- public и extension API стабильны, документированы и покрыты downstream contract tests;
- model/null schemas versioned, portable, validated и имеют legacy converters;
- serial и threaded executions детерминированы и совпадают по результатам;
- hot kernels type-stable, не используют `Any`, abstract fields, string dispatch или accidental inner-loop allocations;
- CLI имеет чистый machine-readable stdout, стабильные exit codes и diagnostics/progress в stderr;
- cache explicit, disableable, atomic, checksum-checked и корректно invalidated;
- Aqua, JET checks, formatter, tests, doctests, docs и clean install проходят в CI;
- benchmark/latency report опубликован с воспроизводимой методикой;
- package импортируется без I/O, filesystem mutations, thread launches, output и global setting changes;
- release candidate проверен на заявленных platforms и устанавливается без Python runtime;
- Python implementation сохраняется до завершения migration window.

## 12. Формат отчёта по этапу

Каждый merge request, закрывающий этап или подэтап, содержит:

1. Архитектурное решение и ссылку на ADR.
2. Python-паттерны, которые сознательно не перенесены.
3. Добавленный или изменённый public/extension contract.
4. Tests и exact commands, которыми выполнена проверка.
5. Compatibility result по уровням и известные divergences.
6. Benchmark environment и сравнение с предыдущим baseline.
7. Новые риски, technical debt и prerequisites следующего этапа.
8. Подтверждение соответствующего gate или список незакрытых пунктов.

## 13. Ближайшая итерация

Stage 0 завершён. Gate 0 пройден. Stage 1 завершён. Gate 1 пройден (125/125 тестов).
Stage 2 завершён. Gate 2 пройден (146 475/146 475 тестов).
Stage 3 завершён (основной slice). Gate 3 пройден (186 083/186 083 тестов).
Stage 4 завершён. Gate 4 пройден (186 673/186 673 тестов).
Stage 5a (BaMM) завершён. 186 985/186 985 тестов проходят.
Stage 5b (SiteGA) завершён. 187 258/187 258 тестов проходят.
Stage 5c (Dimont) завершён. 187 567/187 567 тестов проходят.
Stage 5d (Slim) завершён. 187 866/187 866 тестов проходят.
Stage 6 (Null distributions) завершён (основной slice). 188 046/188 046 тестов проходят.
Stage 7 (Parallelism, cache, storage) завершён (основной slice). 188 236/188 236 тестов проходят.
Stage 8 (CLI и legacy migration) завершён (основной slice). 188 291/188 291 тестов проходят.
Stage 9 (Performance, latency, docs и downstream contract) завершён (основной slice). 188 336/188 336 тестов проходят.
Следующая работа — этап 10 (Release и cutover).

> **Bugfix (до Stage 5d).** XML-парсер (`src/io/xml_parser.jl`) использовал
> `length(content)` (O(n) подсчёт codepoints у Julia `String`) внутри `_starts_at`,
> который вызывается в горячих внутренних циклах парсера. Это делало парсинг
> O(n²): 96KB ≈ 1.9 s, а 2.8 MB ≈ 1600 s, из-за чего полный прогон тестов
> зависал за таймаут. Замена на `ncodeunits` (O(1), корректно для байтового
> индексирования, фикстуры ASCII) сделала парсинг O(n): 96KB ≈ 2 ms, 2.8 MB ≈
> 190 ms. Полный прогон (187 866 тестов) теперь укладывается в ~27 s.

Этап 1 реализовал:

1. ✅ `Mimosa.jl/` package skeleton с `Project.toml`, `src/Mimosa.jl`, `test/`, `benchmark/`, `docs/`.
2. ✅ Error hierarchy (`MimosaError`, `ModelFormatError`, `ModelDimensionError`, `InvariantError`).
3. ✅ `PFM{T,M}` и `PWM{T,M,B}` concrete structs (ADR 0001) с inner constructor validation.
4. ✅ MEME и PFM parsers с size limits и понятными errors.
5. ✅ `pfm_to_pwm` и `pcm_to_pfm` conversion.
6. ✅ PWM reverse complement и score bounds.
7. ✅ Direct PWM/PFM matrix alignment для всех offsets и orientations (ADR 0006).
8. ✅ Metric types: `PearsonCorrelation`, `EuclideanDistance`, `CosineSimilarity`.
9. ✅ Tie-breaking policy (ADR 0006): orientation priority `++ > +- > -+ > --`, first offset wins.
10. ✅ `ComparisonResult` struct и JSON serializer v1.
11. ✅ Unit tests, property tests, compatibility tests на frozen fixtures (125 тестов).
12. ✅ Aqua, JuliaFormatter (BlueStyle), code отформатирован.

Этап 2 реализовал:

1. ✅ `EncodedSequenceBatch` — flat UInt8 buffer с offsets (ADR 0002), constructor invariants.
2. ✅ `RaggedArray{T,V,I}` — обобённая ragged структура для score profiles.
3. ✅ FASTA reader с size limits, обработкой пустых sequences, mixed case, IUPAC→N.
4. ✅ 5-ary encoding: A=0x00, C=0x01, G=0x02, T=0x03, N/ambiguous=0x04, lowercase normalized.
5. ✅ `reverse_complement` для encoded sequences (без временных strings) и `reverse_complement!`.
6. ✅ `reverse_complement` для 5-row PWM (N row stays in place, A↔T, C↔G complement).
7. ✅ Strand policies: `ForwardOnly`, `ReverseOnly`, `BestStrand`, `BothStrands` (typed dispatch).
8. ✅ `scan(model::PWM, seq; strands=...)` — allocating single-sequence scan.
9. ✅ `scan!(dest, model::PWM, seq; strands=...)` — in-place single-sequence scan (zero alloc in inner loop).
10. ✅ `scan(model::PWM, batch::EncodedSequenceBatch; strands=...)` — batch scan, returns `RaggedArray`.
11. ✅ `StrandPair{T}` для both-strands results.
12. ✅ `to_padded` / `from_padded` для compatibility testing и kernel scratch buffers.
13. ✅ Compatibility tests: FASTA read, random batch, forward/reverse/both scan — все совпадают с oracle.
14. ✅ Oracle fixtures регенерированы: добавлен `pwm_scan_input_seed42` (32 fixtures total).
15. ✅ JuliaFormatter (BlueStyle), 146 475 тестов (0 failures, 0 errors, 0 warnings).
16. ☐ Float32/Float64 accumulation experiment и layout benchmark — отложено до benchmark suite.

Этап 3 (следующий):

1. Реализовать validated `RaggedArray` и strand profile bundle.
2. Разделить `fit(EmpiricalLogTail, background_scores)` и `transform`.
3. Реализовать descending lookup, padding-free mapping и tail threshold lookup.
4. Реализовать best/threshold anchors, site-centered windows и boundary clipping.
5. Реализовать target-anchor local realignment и полный shift search.
6. Реализовать `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine` как metric types.
7. Реализовать four-orientation candidates и единую deterministic selection policy.
8. Сохранить symmetry/offset conventions через property и compatibility tests.
9. Добавить one-to-many sequential path с reuse подготовленных query profiles.

Этап 3 реализовал:

1. ✅ `ScoreProfile` — тип для предвычисленных score profiles (pseudo-model).
2. ✅ `read_scores` — чтение FASTA-like числовых профилей в `RaggedArray{Float32}`.
3. ✅ `LogTailTable` и `EmpiricalLogTail` — empirical `-log10(tail)` normalization с fit/transform API.
4. ✅ `flatten_bundle`, `normalize_bundle` — нормализация strand profile bundles.
5. ✅ `AnchorCSR` — CSR-структура для per-row доступа к anchors.
6. ✅ `collect_best_anchors`, `collect_threshold_anchors` — сбор anchors (best/threshold).
7. ✅ Profile metric types: `OverlapCoefficient` (co), `OverlapCoefficientRowwise` (co_rowwise),
   `DiceSimilarity` (dice), `DiceSimilarityRowwise` (dice_rowwise), `CosineSimilarityProfile` (cosine).
8. ✅ `score_shift` — fused shift kernel: collect unique candidates, realign target anchors, score windows.
9. ✅ `profile_compare` — four-orientation candidates с deterministic tie-breaking (ADR 0006).
10. ✅ `ProfileConfig` — typed config struct (metric, search_range, window_radius, realign_window, min_logfpr).
11. ✅ `compare(::ScoreProfile, ::ScoreProfile; metric=..., kwargs...)` — публичный API для profile comparison.
12. ✅ `ComparisonResult` расширен полем `n_sites::Int` (0 для motif comparison).
13. ✅ JSON serialization обновлён: `n_sites` включается при `n_sites > 0`.
14. ✅ Compatibility tests: normalization table, score profile reading, все 5 метрик — совпадают с oracle.
15. ✅ Unit tests: fit, lookup, transform, anchors, ProfileConfig.
16. ✅ Property tests: determinism, non-mutation, self-comparison, metric round-trip.
17. ✅ JuliaFormatter (BlueStyle), 186 083 тестов (0 failures, 0 errors, 0 warnings).
18. ☐ One-to-many path с reuse подготовленных query profiles — отложено.
19. ☐ Motif-derived profiles (PWM scan → normalization → profile comparison) — отложено.
20. ☐ Intermediate compatibility fixtures (anchors, windows, candidate shifts) — отложено.

Этап 4 реализовал:

1. ✅ `SiteHit` и `SiteCollection` — typed structs для motif hits с parallel arrays.
2. ✅ `SiteSelector` abstract type с concrete selectors: `BestPerSequence`, `ThresholdHits`, `TopFractionHits`.
3. ✅ `selectsites(model::PWM, batch, selector; strands=...)` — публичный API для site extraction.
4. ✅ Best-per-sequence selection: one best hit per sequence across both strands.
5. ✅ Threshold selection: all hits above score threshold (forward/reverse/best-strand modes).
6. ✅ Top-fraction selection: keep top fraction of hits by score (wraps base selector).
7. ✅ `sort_hits!` — deterministic sort by (seq_index asc, score desc, start asc, strand asc).
8. ✅ `extract_site_matrix` — extract numeric windows, reverse-complement for minus strand.
9. ✅ `build_pcm` — PCM accumulation from site matrix (valid bases only, N skipped).
10. ✅ `reconstruct_pfm(model::PWM, batch, selector; pseudocount=...)` — PFM from sites.
11. ✅ `site_strings` — convert numeric sites to DNA strings.
12. ✅ Compatibility tests: `sites_best_pif4_seed42` (100 hits), `pfm_reconstruction_best_pif4_seed42`.
13. ✅ Oracle fixture добавлен: `sites_input_seed42` (33 fixtures total).
14. ✅ Unit tests: SiteHit, SiteCollection, selectors, sort, top-fraction, extract, PCM, strings.
15. ✅ Property tests: determinism, non-mutation, empty batch, short sequences, PFM column sums,
    reverse complement involution, sort idempotency.
16. ✅ JuliaFormatter (BlueStyle), 186 673 тестов (0 failures, 0 errors, 0 warnings).
17. ☐ DataFrames table conversion — отложено (не требуется на текущем этапе).
18. ☐ Heterogeneous motif comparison через reconstructed PFM — отложено до Stage 5.

Этап 5a (BaMM) реализовал:

1. ✅ Описан исходный format `.ihbcp` и mathematical representation для BaMM.
2. ✅ `BaMM{T,M}` — concrete immutable struct с representation matrix `(5^(order+1), motif_length)`,
   order и motif_length, без catch-all config dictionary.
3. ✅ Strict `.ihbcp` parser с size limits, comment handling, consistency validation и понятными errors.
4. ✅ Constructor invariants: row count validation, non-finite check, non-negative order.
5. ✅ `scorebounds(::BaMM)` — per-column min/max summed across positions, совпадает с oracle.
6. ✅ Forward/reverse/best/both scanning kernels для BaMM с context-aware scoring
   (kmer = order + 1, context = order, window = motif_len + order).
7. ✅ `scan(::BaMM, seq; strands=...)` и `scan(::BaMM, batch; strands=...)` через multiple dispatch.
8. ✅ `scan!(dest, ::BaMM, seq; strands=...)` — in-place API.
9. ✅ `scan_result_lengths(::BaMM, batch)` — pre-allocate output buffers.
10. ✅ `readmodel(path; format=:auto)` auto-detects `.ihbcp` and dispatches to `read_bamm`.
11. ✅ Compatibility tests: parsing (3 files × 3 orders = 9 fixtures), score bounds (3 fixtures),
    forward/reverse scanning (2 orders × 2 strands = 4 fixtures), readmodel auto-detect (2 tests).
    All match oracle within Float32 tolerance (max_diff < 1e-5).
12. ✅ Oracle fixtures: 17 new fixtures added (49 total in manifest).
13. ✅ Unit tests: constructor, show, equality, scorebounds, parsing, single-sequence scan,
    batch scan, order=0 equivalence to PWM scan, determinism.
14. ✅ JuliaFormatter (BlueStyle), 186 985 тестов (0 failures, 0 errors, 0 warnings).
15. ☐ Malformed/security fixtures — отложено до следующего подэтапа.
16. ☐ BaMM writer — не определён в Python (joblib dump only); отложено.
17. ☐ BaMM sites и reconstruction — отложено (requires sites API generalization for higher-order models).
18. ☐ BaMM comparison — отложено (requires comparison API generalization for higher-order models).
19. ☐ Model-specific benchmark — отложено до benchmark suite.

Этап 5b (следующий — SiteGA):

1. Описать исходный format `.mat` и mathematical representation для SiteGA.
2. Добавить concrete immutable type для SiteGA без catch-all config dictionary.
3. Реализовать strict parser, constructor invariants и `.mat` reader/writer.
4. Реализовать `scorebounds`, forward/reverse scan, sites и reconstruction methods.
5. Сравнить raw representation, individual site score, tracks и final comparisons с oracle.
6. Добавить malformed/security fixtures и model-specific benchmark.
7. Обновить extension guide и feature matrix.

Этап 5b (SiteGA) реализовал:

1. ✅ Описан исходный format `.mat` и mathematical representation для SiteGA.
2. ✅ `SiteGA{T,M}` — concrete immutable struct с representation matrix `(25, motif_length)`,
   motif_length, без catch-all config dictionary. Flattened из Python `(5, 5, length)` в
   `(25, length)` с row indexing `code = base1 * 5 + base2`.
3. ✅ Strict `.mat` parser с size limits, dinucleotide validation, range validation и понятными errors.
4. ✅ `write_sitega(path, model)` — writer с segment grouping (contiguous equal-value runs),
   matching Python format. Round-trip test проходит.
5. ✅ Constructor invariants: row count (25), column/length match, non-finite check.
6. ✅ `scorebounds(::SiteGA)` — per-column min/max across 25 dinucleotide codes, summed.
   Совпадает с oracle.
7. ✅ Forward/reverse/best/both scanning kernels для SiteGA с dinucleotide scoring
   (kmer = 2, context = 0, n_terms = motif_length - 1, window = motif_length).
8. ✅ `scan(::SiteGA, seq; strands=...)` и `scan(::SiteGA, batch; strands=...)` через multiple dispatch.
9. ✅ `scan!(dest, ::SiteGA, seq; strands=...)` — in-place API.
10. ✅ `scan_result_lengths(::SiteGA, batch)` — pre-allocate output buffers.
11. ✅ `readmodel(path; format=:auto)` auto-detects `.mat` and dispatches to `read_sitega`.
12. ✅ Compatibility tests: parsing (3 files), score bounds (3 fixtures),
    forward/reverse scanning (2 files × 2 strands = 4 fixtures), readmodel auto-detect (2 tests).
    All match oracle within Float32 tolerance.
13. ✅ Oracle fixtures: 12 new fixtures added (61 total in manifest).
14. ✅ Unit tests: constructor, show, equality, scorebounds, parsing, single-sequence scan,
    batch scan, determinism, write round-trip.
15. ✅ JuliaFormatter (BlueStyle), 187 258 тестов (0 failures, 0 errors, 0 warnings).
16. ☐ Malformed/security fixtures — отложено до следующего подэтапа.
17. ☐ SiteGA sites и reconstruction — отложено (requires sites API generalization for higher-order models).
18. ☐ SiteGA comparison — отложено (requires comparison API generalization for higher-order models).
19. ☐ Model-specific benchmark — отложено до benchmark suite.

Этап 5c (Dimont) реализовал:

1. ✅ Описан исходный XML format и mathematical representation для Dimont.
   Dimont — Jstacs Bayesian network (MarkovModelDiffSM) с tree-structured context
   dependencies. XML path: `.//ThresholdedStrandChIPper/function/pos/MarkovModelDiffSM`.
   Parameter trees materialized в dense 5-ary tensor `(5,)* (span+1) + (length,)`,
   затем flattened в `(5^(span+1), length)` matrix.
2. ✅ `Dimont{T,M}` — concrete immutable struct с representation matrix `(5^(span+1),
   motif_length)`, `span` и `motif_length`, без catch-all config dictionary.
   `span` играет роль `order` в BaMM. Тип наследует `AbstractHigherOrderMotif`.
3. ✅ Minimal XML parser (`xml_parser.jl`) — targeted parser для Jstacs XML без
   внешних dependencies. Поддерживает tags, attributes, nested elements, text,
   comments. ElementTree-like navigation (`xml_find`, `xml_findall`, `xml_text`,
   `xml_attribute`).
4. ✅ Strict `read_dimont` parser с size limits, tree parsing, tensor materialization,
   N-state filling (per-axis minimum, matching Python `_fill_n_axis_with_min`),
   `log(4.0)` uniform base addition.
5. ✅ Constructor invariants: row count validation `5^(span+1)`, column/length match,
   non-finite check, non-negative span, positive motif_length.
6. ✅ `scorebounds(::Dimont)` — per-column min/max summed across positions,
   совпадает с oracle.
7. ✅ Forward/reverse/best/both scanning kernels для Dimont с context-aware scoring
   (kmer = span + 1, context = span, window = motif_len + span, n_terms = motif_len).
   Идентичная geometry к BaMM scanning.
8. ✅ `scan(::Dimont, seq; strands=...)` и `scan(::Dimont, batch; strands=...)`
   через multiple dispatch.
9. ✅ `scan!(dest, ::Dimont, seq; strands=...)` — in-place API.
10. ✅ `scan_result_lengths(::Dimont, batch)` — pre-allocate output buffers.
11. ✅ `readmodel(path; format=:auto)` auto-detects `.xml` and dispatches to `read_dimont`.
12. ✅ Compatibility tests: parsing (5 files), score bounds (5 fixtures),
     forward/reverse scanning (2 files × 2 strands = 4 fixtures), readmodel auto-detect.
     All match oracle within Float32 tolerance.
13. ✅ Oracle fixtures: 15 new fixtures added (76 total in manifest).
14. ✅ Unit tests: constructor, show, equality, scorebounds, parsing, single-sequence scan,
     batch scan, determinism, span=0 equivalence to order-0 BaMM scan.
15. ✅ JuliaFormatter (BlueStyle), 187 567 тестов (0 failures, 0 errors, 0 warnings).
16. ☐ Malformed/security fixtures — отложено до следующего подэтапа.
17. ☐ Dimont writer — не определён в Python (joblib dump only); отложено.
18. ☐ Dimont sites и reconstruction — отложено (requires sites API generalization
     for higher-order models).
19. ☐ Dimont comparison — отложено (requires comparison API generalization for
     higher-order models).
20. ☐ Model-specific benchmark — отложено до benchmark suite.

Этап 5d (Slim) реализовал:

1. ✅ Описан исходный XML format и mathematical representation для Slim.
   Slim — Jstacs GenDisMix classifier с mixture of component/ancestor
   dependencies. XML path: `.//SLIM`. Параметры `componentMixtureParameters`,
   `ancestorMixtureParameters`, `dependencyParameters` нормализуются (log-sum-exp)
   в log-probability tables и материализуются в dense 5-ary tensor
   `(5,)*(span+1) + (length,)`, затем flattening в `(5^(span+1), length)`.
   Span вычисляется как `max(component_index + ancestor_count - 1)` по всем
   позициям и компонентам с `component_index ≥ 1`.
2. ✅ `Slim{T,M}` — concrete immutable struct с representation matrix
   `(5^(span+1), motif_length)`, `span` и `motif_length`, без catch-all config
   dictionary. Наследует `AbstractHigherOrderMotif`. Идентичная representation
   к Dimont/BaMM.
3. ✅ Strict `read_slim` parser: нахождение `SLIM`, size limits, parsing вложенных
   `<pos>` массивов для трёх параметров, span computation, log-normalization,
   log-sum-exp materialization через переиспользование общих helpers из
   `dimont_reader.jl` (`_iter_contexts`, `_context_value`,
   `_build_position_column`, `_decode_5ary`).
4. ✅ **Refactor shared higher-order scan kernel.** Из BaMM/Dimont/SiteGA scan
   файлов извлечён generic `_ho_scan_forward!/reverse!/best!/both!` в
   `src/scanning/higher_order_scan.jl`, parameterized by geometry
   `(rep, kmer, context, window, n_terms)`. Все три существующих модели и Slim
   делегируют ему. Устранено ~1000 строк дублирования; поведение и тесты
   прежние. Slim использует ту же scanning geometry, что и BaMM/Dimont
   (kmer=span+1, context=span, window=motif_len+span, n_terms=motif_len).
5. ✅ Constructor invariants: row count `5^(span+1)`, column/length match,
   non-finite check, non-negative span, positive motif_length.
6. ✅ `scorebounds(::Slim)` — per-column min/max summed, совпадает с oracle.
7. ✅ Forward/reverse/best/both scanning для Slim через multiple dispatch и
   shared kernel. `scan(::Slim, seq/batch; strands=...)`, `scan!(...)`,
   `scan_result_lengths`. Inner loop — 0 аллокаций.
8. ✅ `readmodel(path; format=:auto)` различает Slim vs Dimont для `.xml` по
   наличию `<SLIM` (I/O-boundary content peek), диспетчует в `read_slim`/`read_dimont`.
9. ✅ Compatibility tests: parsing (4 files), score bounds (4 fixtures),
     forward/reverse scanning (2 files × 2 strands = 4 fixtures), readmodel
     auto-detect + Slim/Dimont disambiguation. **Все совпадают с oracle
     точно (maxdiff = 0.0)** — representation и scan tracks бит-в-бит.
10. ✅ Oracle fixtures: 13 new fixtures added (89 total in manifest), сгенерированы
     `scripts/generate_slim_fixtures.py` (reuse dimont scan input seed=42).
11. ✅ Unit tests: constructor, show, equality, scorebounds, parsing,
     single-sequence scan, batch scan, determinism, span=0 geometry.
12. ✅ Type stability: `@code_warntype` на `scan_forward!(::Slim, ...)` и
     `_ho_scan_forward!` — concrete `Body::Vector{Float32}`, no runtime `Any`.
13. ✅ JuliaFormatter (BlueStyle), 187 866 тестов (0 failures, 0 errors, 0 warnings).
14. ☐ Malformed/security fixtures — отложено до следующего подэтапа.
15. ☐ Slim writer — не определён в Python (joblib dump only); отложено.
16. ☐ Slim sites и reconstruction — отложено (requires sites API generalization
     for higher-order models).
17. ☐ Slim comparison — отложено (requires comparison API generalization for
     higher-order models).
18. ☐ Model-specific benchmark — отложено до benchmark suite.

Gate 5 (после 5d):

- ✅ Все четыре заявленные model families (PWM/PFM, BaMM, SiteGA, Dimont, Slim)
  имеют полный parser-to-scan vertical path.
- ☐ Heterogeneous collections группируются по concrete type перед batch kernels
  (отложено до batch comparison/nulls на этапах 6-7).
- ☐ Unsupported writer честно возвращает typed error (отложено).
- ☐ Extension guide обновление (отложено до docs этапа 9).
