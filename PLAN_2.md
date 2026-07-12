# План закрытия проблем Mimosa.jl перед release candidate

## 1. Назначение и статус документа

Этот документ дополняет `PLAN.md` и `REFACTORING.md` результатами аудита текущей реализации `Mimosa.jl`.
Он не заменяет исходный план миграции и не меняет научные контракты. При конфликте источники истины имеют следующий
приоритет:

1. `REFACTORING.md` — архитектурные и научные требования;
2. `PLAN.md` — этапы миграции, compatibility policy и общий Definition of Done;
3. `PLAN_2.md` — remediation backlog для закрытия обнаруженных проблем;
4. `AGENTS.md` — практические инструкции, которые должны обновляться после прохождения gate.

Аудит выполнен по рабочему дереву 12 июля 2026 года. На Julia 1.12.6 с четырьмя threads команда
`julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'` завершилась успешно: `188411/188411` тестов прошли. Проверка
JuliaFormatter для `Mimosa.jl/src` и `Mimosa.jl/test` также прошла. Эти результаты подтверждают зрелость реализованных
happy paths, но не закрывают перечисленные ниже функциональные, security, CI, documentation и release gaps.

## 2. Цель работ

Цель — довести пакет до проверяемого release candidate, для которого одновременно выполняются условия:

- CLI не может сообщить конфигурацию, отличающуюся от реально выполненного алгоритма;
- untrusted input не позволяет выйти за пределы bundle, отключить checksum validation или вызвать unsafe indexing;
- Julia correctness и quality gates обязательны в CI на заявленных версиях и платформах;
- feature matrix, README, API и CLI documentation соответствуют фактическому поведению;
- benchmark и latency report воспроизводимы и содержат все данные, требуемые `PLAN.md`;
- downstream contract проверяется отдельным consumer environment;
- package metadata и release artifacts готовы к регистрации и clean-room установке;
- рефакторинг дублирования выполняется только после закрытия correctness и compatibility blockers.

## 3. Сводка проблем и приоритетов

| ID | Приоритет | Область | Проблема | Блокирует |
|---|---|---|---|---|
| P0-1 | critical | Null/CLI | `build-null` разбирает, но не передаёт strategy, metric и большую часть options | Gate 8, RC |
| P0-2 | critical | Null/API | `build_null` не выбирает алгоритм по strategy и хранит `metric::Any` | Gate 6, Gate 8 |
| P0-3 | critical | Storage | Bundle filename допускает path traversal; null checksum можно фактически пропустить | Gate 7, RC |
| P0-4 | critical | Sequences | Публичный encoded batch допускает коды вне `0:4`, используемые под `@inbounds` | Gate 2, RC |
| P1-1 | high | CI | GitHub Actions не запускает Julia вообще | Gate 9, Gate 10 |
| P1-2 | high | Documentation | CLI и quick-start examples не соответствуют API и не исполняются | Gate 9 |
| P1-3 | high | Audit | `docs/feature_matrix.md` остался в состоянии Stage 0/1 | Definition of Done |
| P1-4 | high | Quality gates | Aqua, Documenter и precompile workload могут скрывать ошибки | Gate 9 |
| P1-5 | high | Package | Julia compat и runtime dependencies заданы некорректно | Gate 10 |
| P1-6 | high | Public API | Downstream test не является отдельным consumer package; API покрыт частично | Gate 9 |
| P2-1 | medium | Models | PFM/PWM/background и higher-order constructors не полностью валидируют invariants | RC hardening |
| P2-2 | medium | Performance | Нет полного latency/RSS/precompile report и regression baseline | Gate 9 |
| P2-3 | medium | Cache/nulls | Cache не интегрирован в workflows; нет null compatibility lookup | Feature parity |
| P2-4 | medium | CLI | Нет subprocess coverage и части Python-facing options | Gate 8, parity |
| P3-1 | low | Maintainability | Дублируются adapters higher-order scanning | После correctness |
| P3-2 | low | Maintainability | `cli.jl` объединяет parser, validation и orchestration | После API freeze |

## 4. Порядок выполнения

Работы выполняются последовательно по gate. Задачи внутри одного этапа можно вести независимо только если они не
меняют одни и те же public contracts или frozen compatibility semantics.

```text
A. Scientific correctness and unsafe-input blockers
  -> B. Storage and constructor hardening
      -> C. Mandatory Julia CI and contract enforcement
          -> D. Documentation and feature inventory repair
              -> E. Performance, latency and release evidence
                  -> F. Release candidate and cutover decision
                      -> G. Optional duplication reduction
```

## 5. Этап A. Scientific correctness и CLI contracts

### A1. Исправить typed null build API

**Статус: реализовано 12 июля 2026 года.** Введены typed policies `MotifNullStrategy` и
`ProfileNullStrategy`, `NullBuildConfig` проверяет совместимость strategy/metric при любом публичном
конструировании, а profile path требует явный `EncodedSequenceBatch`. Metadata использует фактические model,
relation, sequence и background fingerprints. Frozen compatibility coverage остаётся обязательной частью Gate R1.

**Проблема:** `NullBuildConfig.metric` имеет тип `Any`; `strategy` хранится строкой; `build_null` всегда вызывает
двухаргументный `compare(q, t)` независимо от strategy.

**Работы:**

- заменить строковый algorithm switch на typed policy, например `MotifNullStrategy` и `ProfileNullStrategy`, либо на
  отдельные методы `build_null` с typed configuration;
- параметризовать `NullBuildConfig` конкретным типом metric вместо поля `Any`;
- валидировать metric на API boundary, а не только в CLI;
- для motif strategy выполнять direct motif comparison;
- для profile strategy явно принимать `EncodedSequenceBatch`, optional background batch и `ProfileConfig`;
- исключить fallback, при котором неизвестный strategy сохраняется в metadata, но выполняется другой алгоритм;
- заполнить `model_collection_fingerprint`, `relation_fingerprint`, `sequence_fingerprint` и
  `background_fingerprint` фактическими значениями;
- оформить ADR, если меняется публичная сигнатура или schema metadata.

**Тесты:**

- unit: motif strategy с каждой motif metric;
- unit: profile strategy с каждой profile metric;
- unit: invalid strategy/metric/config завершается до начала вычислений;
- unit: изменение sequences/background меняет соответствующий fingerprint;
- property: serial и threaded raw scores, pair order и fitted result совпадают;
- compatibility: raw pair schedule, raw scores и GEV parameters против frozen Python fixtures;
- regression: profile/co и motif/pcc дают разные ожидаемые результаты на одном corpus.

**Критерий закрытия:** невозможно создать `NullDistribution`, metadata которого не соответствует реально выполненной
strategy, metric и входным данным.

### A2. Исправить `build-null` CLI wiring

**Статус: реализовано 12 июля 2026 года.** CLI передаёт parsed relation, strict, null-build и profile parameters
в typed API; profile strategy использует FASTA либо seeded random sequences, а summary строится из фактического
`NullDistribution`. Добавлены integration checks для motif и profile strategy. Subprocess coverage остаётся
обязательной частью Gate R1.

**Проблема:** CLI разбирает `--strategy`, `--metric`, `--fasta`, `--seed`, `--num-sequences`, `--seq-length`, profile
alignment options, `--strict`, `--min-null-targets`, `--name-column`, `--group-column` и `--ignore-missing`, но не
передаёт их полностью в library API.

**Работы:**

- ввести typed `BuildNullCommand` или аналогичную immutable config после parsing;
- передавать все задокументированные options в public API;
- читать FASTA или генерировать sequences только для profile strategy;
- использовать `name-column`, `group-column` и `ignore-missing` при чтении relations;
- передавать `metric`, `strict`, `min_null_targets` и profile parameters;
- выводить summary только из фактической `NullDistribution`, а не из исходных строк CLI;
- проверить `tryparse` results для всех integer/float options и диапазоны значений;
- либо реализовать `--quiet`/`--verbose`, либо удалить ложные promises из help до реализации.

**Тесты:**

- subprocess test для motif strategy и non-default metric;
- subprocess test для profile strategy с FASTA;
- option-effect tests: каждое значимое CLI option изменяет config/result или выдаёт controlled error;
- test на `--strict` и `--min-null-targets`;
- JSON summary сравнивается с сохранённым manifest;
- malformed numeric values возвращают usage exit code без stacktrace.

**Критерий закрытия:** для каждой принятой CLI option существует тест, доказывающий её влияние или корректную
валидацию; summary и сохранённый bundle согласованы.

### A3. Завершить CLI statistical annotation contract

**Статус: реализовано 12 июля 2026 года.** `motif` и `profile` принимают
`--pvalue`, явный `--null-distribution` и `--effective-number-of-targets`.
Перед annotation CLI проверяет strategy, metric и sequence/background fingerprints,
а annotated JSON маркируется `annotation_schema_version = 1`. Автоматический поиск
null bundles и workflow cache сознательно остаются deferred; global cache не переносится.

**Работы:**

- сверить Python CLI feature set для `--pvalue`, `--null-distribution`, cache options и output handling;
- реализовать отсутствующие обязательные options либо пометить `not-porting`/`deferred` с обоснованием в feature matrix;
- проверять compatibility metadata null distribution до annotation;
- не выполнять автоматический поиск в глобальном cache, если он сознательно не переносится;
- зафиксировать JSON schema version для annotated results.

**Критерий закрытия:** все Python user-facing CLI capabilities имеют один из согласованных статусов `done`,
`documented-divergence`, `deferred` или `not-porting`; статуса `planned` без владельца и этапа нет.

## 6. Этап B. Security и invariant hardening

### B1. Защитить portable model/null bundles

**Статус: реализовано 12 июля 2026 года.** Model и null storage используют общий
bounded TOML/NPY boundary: v1 manifest и checksum обязательны и типизированы,
пути проверяются до `realpath` и не могут выйти из bundle root, NPY headers и
payload length разбираются строго до allocation, а model-specific shape/order/span
инварианты проверяются до чтения blob. Запись собирается в sibling staging
directory и коммитится одной rename-операцией; orphan stages не читаются.
Hostile tests покрывают traversal, symlink escape, checksum/version/type/size
violations, malformed NPY и staged-write cleanup.

**Работы:**

- разрешать в manifest только относительные basename либо нормализованные пути внутри bundle root;
- запрещать absolute paths, `..`, symlink escape и platform-specific traversal variants;
- требовать checksum с точным форматом `sha256:<64 lowercase hex>`;
- считать отсутствующий/неизвестный checksum format ошибкой, а не пропускать validation;
- установить limits для manifest, каждого blob, числа arrays, dimensions и общего allocation budget;
- строго разбирать NPY header: magic, version, header length, dtype, endianness, rank, shape и payload length;
- проверять, что manifest shape/dtype совпадают с NPY и model constructor invariants;
- принимать только поддерживаемые положительные format versions; version `0`, negative и non-integer отклонять;
- сохранять atomicity manifest и blobs; определить recovery policy для orphan temp files;
- использовать typed `ModelFormatError`/`InvariantError` вместо случайных `KeyError`, `BoundsError` и `error()`.

**Security tests:**

- `../outside.npy`, absolute filename и symlink escape;
- missing, malformed и mismatched checksum;
- oversized manifest/blob/dimensions;
- truncated header/payload, wrong dtype/rank/endianness;
- missing keys, wrong TOML types и unsupported versions;
- partial write и corrupted cache/bundle behavior;
- проверка, что ни один тест не читает файл за пределами временного bundle root.

**Критерий закрытия:** hostile bundle corpus выдаёт только controlled typed errors и не читает данные за пределами
bundle root; валидные v1 bundles сохраняют round-trip compatibility.

### B2. Валидировать encoded sequences до unsafe kernels

**Статус: реализовано.** Все публичные конструкторы `EncodedSequenceBatch`
проверяют `0 <= code <= N_CODE`; внутренний unsafe-конструктор
`_unsafe_encoded_batch` использует `Val{:unsafe}` токен для hot paths
(`make_random_sequences`). `from_padded` валидирует padding, lengths и codes.
`reverse_complement!` проверяет aliasing dest/src. Scan kernels
(`scan_forward!`, `scan_reverse!`, `scan_best!`, `scan_both!`, и все
`_ho_scan_*!` kernels) валидируют `n_pos >= 0` и destination size до входа
в `@inbounds`. `extract_site_matrix` проверяет, что site window не выходит
за пределы sequence. Инварианты документированы рядом с каждым
`@inbounds` kernel. Добавлен `test/unit/test_validation.jl` с 237 тестами,
покрывающими invalid codes, aliasing, short destinations, empty/short
sequences, fuzzed inputs, allocating/in-place equivalence и все model
constructor invariants (B3).

**Работы:**

- гарантировать `0 <= code <= N_CODE` во всех публичных constructors и conversion APIs;
- решить, должен ли raw constructor валидировать всегда или unsafe construction станет internal-only;
- валидировать destination sizes перед входом в `@inbounds` scanning kernels;
- проверить aliasing rules для in-place reverse complement и scan destinations;
- документировать invariants непосредственно рядом с каждым approved `@inbounds` kernel.

**Тесты:** invalid code `0x05`/`0xff`, short destination, invalid offsets, empty rows, short sequences, allocating/in-place
equivalence и fuzzed encoded inputs.

**Критерий закрытия:** любой public input либо удовлетворяет kernel invariants, либо отклоняется до `@inbounds` участка.

### B3. Усилить model constructors

**Статус: реализовано.** PFM теперь проверяет 4 строки, положительную width,
finite и non-negative values. PWM проверяет background на finite,
non-negative и сумму ~1.0 (rtol=1e-4). BaMM/SiteGA/Dimont/Slim проверяют
order/span >= 0 и <= 10 (guard против exponentiation blow-up) до
вычисления 5^(order+1). Все constructors проверяют finite values.

**Работы:**

- PFM: проверить 4 строки, положительную width, finite и non-negative values, а также согласованную policy для column sums;
- PWM: проверить background на finite, positive/non-negative policy и сумму с утверждённой tolerance;
- BaMM/SiteGA/Dimont/Slim: проверить order/span до exponentiation и allocation-sensitive calculations;
- добавить верхние limits в file readers, не ограничивая обычное in-memory scientific API без ADR;
- унифицировать typed errors и сообщения без пустого path, где input создан в памяти.

**Критерий закрытия:** constructors действительно обеспечивают invariants, заявленные в docstrings и используемые
unsafe kernels.

## 7. Этап C. Обязательный Julia CI и quality gates

### C1. Добавить Julia CI matrix

Минимальная обязательная matrix:

- Linux x86_64: minimum supported Julia и latest stable;
- macOS arm64: latest stable до RC;
- Julia nightly: allowed failure до отдельного support decision;
- `JULIA_NUM_THREADS=1` и многопоточная конфигурация;
- clean `Pkg.instantiate`, `Pkg.precompile`, `using Mimosa` smoke test и `Pkg.test`;
- отдельные formatter, docs/doctests, Aqua, JET targeted и coverage jobs;
- subprocess CLI integration job;
- security corpus job;
- scheduled/manual benchmark job с сохранением artifacts.

CI не должен запускать oracle Python во время обычных Julia tests. Python CI сохраняется отдельно до завершения
migration window.

### C2. Сделать quality checks fail-closed

**Работы:**

- убрать общий `try/catch` вокруг Aqua; отсутствие Aqua и test failure должны падать;
- включить ранее отключённые Aqua checks либо документировать точечные исключения;
- добавить JET как test/dev dependency и targeted checks для public workflows/hot kernels;
- убрать `warnonly=true` из release docs build;
- убрать подавление ошибок representative precompile workload;
- добавить проверку docstring для каждого exported symbol;
- запретить warnings в RC quality job, кроме явно allowlisted platform warnings.

**Критерий закрытия:** намеренно внесённая ошибка в Aqua, doc reference, doctest, precompile workload или JET target
делает соответствующий CI job красным.

### C3. Исправить package environments

**Работы:**

- перенести JuliaFormatter из runtime dependencies в dev/test tooling через `Pkg`;
- проверить stdlib compat на реальной minimum Julia; не требовать stdlib 1.11 при support Julia 1.10;
- определить support policy: Julia 1.10 LTS либо Julia 1.11+, затем согласовать все environments и docs;
- добавить compat для test/docs dependencies;
- проверить отсутствие stale Manifest assumptions в documented commands;
- измерить dependency budget и import/precompile impact.

**Критерий закрытия:** fresh clone устанавливается и тестируется на каждой заявленной Julia version без ручного
`Pkg.resolve` и без Python.

### C4. Сделать downstream contract реальным consumer package

**Работы:**

- создать отдельный `test/downstream/Project.toml` без доступа к test internals;
- импортировать только зарегистрированный/developed `Mimosa` public API;
- покрыть scanning, direct/profile comparison, prepared one-to-many, sites, reconstruction, nulls, storage и annotation;
- запретить обращения к `_internal` names и package source paths;
- проверить contract в отдельном clean CI environment;
- сократить exports после API review: internal helpers и implementation constants не должны экспортироваться без нужды.

**Критерий закрытия:** downstream package проходит независимо и представляет минимальный контракт, реально нужный
MotifHORDE.jl.

## 8. Этап D. Документация и inventory repair

### D1. Полностью обновить feature matrix

Для каждой строки `docs/feature_matrix.md` указать:

- Julia status: `done`, `partial`, `documented-divergence`, `deferred`, `not-porting`;
- public Julia entry point;
- compatibility fixture/test ID;
- известные ограничения;
- owner/stage для незакрытой работы.

Отдельно сверить model families, formats, all strand policies, direct/profile one-to-one и one-to-many, normalization,
sites, reconstruction, null metadata/lookup, cache, CLI options и legacy converters.

### D2. Исправить README, CLI guide и quick start

**Работы:**

- убрать устаревший Stage 1 status;
- использовать реальные positional arguments и обязательные `--model*-type` options;
- исправить `readsequences` destructuring;
- использовать `build_null(...).distribution` там, где требуется `NullDistribution`;
- исправить examples для `pvalue`, `savenull`, cache и conversion commands;
- проверить все указанные example filenames;
- объяснить current registration/install status без обещания несуществующего registry package;
- синхронизировать help text, docs и integration tests из одного declarative source либо добавить consistency test.

### D3. Включить executable documentation

**Работы:**

- превратить quick-start snippets в doctests или integration examples;
- исполнять CLI examples как subprocess в temp directories;
- проверять все `@ref` и `@docs` без warn-only режима;
- добавить docstrings отсутствующим exported symbols либо убрать их из exports;
- публиковать docs только после успешного build/test job.

**Критерий закрытия этапа D:** новый пользователь может выполнить README/quick-start/CLI examples без исправления
аргументов или типов вручную; feature matrix согласована с тестами и текущим кодом.

## 9. Этап E. Performance, latency и maintainability

### E1. Опубликовать воспроизводимый benchmark report

Report должен содержать:

- commit SHA, Julia и Python oracle versions;
- версии всех значимых dependencies;
- CPU model, OS/kernel, RAM, thread count и power/performance mode;
- input sizes, model widths/orders, seeds и dataset/fixture IDs;
- warm-up policy, sample/evaluation counts и cold/warm classification;
- median/min/variance, allocations и peak RSS;
- serial performance и scaling на 1/2/4/available threads;
- Python comparison там, где заявляется migration performance;
- package instantiate/precompile time, `using Mimosa`, first call, repeated call и real CLI subprocess startup;
- raw machine-readable BenchmarkTools output как CI/release artifact.

Добавить representative workloads из `PLAN.md`: ragged heavy-tail batches, BaMM orders 1–5, one-to-many 10/100/1000,
dense threshold anchors, high/low site density и null schedules до practically affordable scale.

### E2. Ввести regression baseline без хрупких shared-runner limits

- хранить baseline по контролируемой машине или сравнивать только устойчивые normalized metrics;
- scheduled CI публикует comparison report, но не блокирует PR по шумным microbenchmarks;
- RC gate блокируется при подтверждённой регрессии согласованных representative workloads;
- любое optimization change сопровождается profile evidence и compatibility rerun.

### E3. Завершить type/allocation audit

- targeted JET и `@code_warntype` для public hot paths, а не только отдельных inner kernels;
- проверить heterogeneous model collections и `build_null` schedule;
- убрать `Any` и abstract pair containers из performance-sensitive orchestration;
- зафиксировать Float32/Float64 accumulation experiment и tolerance rationale;
- измерить layout alternatives, отложенные в Stage 2/3, и обновить ADR/data-layout docs.

### E4. Сократить дублирование после API freeze

**Higher-order scanning:** вынести общие allocating/in-place/batch adapters для BaMM/SiteGA/Dimont/Slim через небольшой
trait/interface: representation, kmer, context length, window size, term count и site offset. Не использовать generated
functions или macros без измеримой необходимости.

**CLI:** разделить parser, typed command configs и command runners. Parser не должен выполнять scientific validation;
runner не должен повторно интерпретировать строки. Не добавлять внешний CLI dependency без latency/maintenance review.

**Storage:** объединить повторяющийся array manifest validation и NPY reading для model/null bundles без создания
универсального `Dict{String,Any}` domain layer.

**Критерий закрытия:** сокращение кода подтверждено неизменными compatibility results и отсутствием ухудшения latency;
рефакторинг не расширяет public API.

## 10. Этап F. Release candidate

### F1. Release metadata и distribution

- добавить `CITATION.cff`, release notes/changelog и проверенные license metadata;
- проверить General registry requirements, package name/UUID/version и compat bounds;
- определить repository URL, docs canonical URL и окончательную topology;
- собрать release candidate tag и проверить install по tag в fresh depot;
- рассмотреть PackageCompiler app отдельно; static binary не блокирует library RC без утверждённого требования;
- документировать conda/Bioconda strategy без добавления Python runtime dependency.

### F2. Platform и clean-room validation

- Linux x86_64 и macOS arm64 обязательны;
- Windows support либо проверяется, либо явно исключается из support promise;
- проверить fresh user depot, read-only working tree и отсутствие локальных absolute paths;
- импорт не создаёт files/directories, не печатает, не запускает threads и не меняет global settings;
- CLI artifact работает без Python;
- rollback означает установку предыдущей package version.

### F3. Migration window

- Python oracle остаётся доступным и pinned;
- определить срок dual-support и владельца compatibility fixes;
- Python implementation объявляется legacy/deprecated только после RC review и downstream acceptance;
- удаление Python code не входит в этот remediation plan.

## 11. Отдельный backlog незавершённых возможностей

Следующие пункты нельзя считать автоматически закрытыми прохождением тестов. Для каждого требуется решение
`implement`, `defer` или `not-porting` с отражением в feature matrix:

- empirical rank-based fallback для failed GEV fit;
- explicit `AbstractRNG` API и thread-count-independent seed derivation там, где библиотека генерирует данные;
- degenerate, constant, tiny, NaN/Inf и extreme-tail statistical corpus;
- null compatibility lookup/search с explicit directories;
- cache integration в scan/profile/compare/build-null hot workflows;
- cross-language model/null bundle exchange fixtures;
- Julia-native `convert-null` command;
- interactive progress и debug stacktrace mode;
- DataFrames extension, только если появляется реальный downstream demand;
- PackageCompiler app/static binary;
- conda/Bioconda packaging.

## 12. Обязательная test matrix по изменениям

| Изменение | Unit | Property | Compatibility | Integration | Security | Benchmark |
|---|---:|---:|---:|---:|---:|---:|
| Null strategy/config | да | да | да | да | — | да |
| CLI option wiring | да | — | да | subprocess | — | latency |
| Bundle reader/storage | да | round-trip | cross-language | да | обязательно | read/write |
| Sequence/model validation | да | fuzz/property | fixture rerun | да | обязательно | kernel unchanged |
| Parallel scheduling | да | serial=threaded | raw order | да | — | scaling |
| Public API/downstream | да | — | — | clean consumer | — | import latency |
| Optimization/refactor | да | да | полный rerun | relevant | relevant | до/после |

Тестовые tolerances не ослабляются ради прохождения. Любое новое numerical divergence требует rationale, regression test
и при необходимости ADR/migration note.

## 13. Verification commands

Минимальный локальный набор перед merge:

```bash
export PATH="$HOME/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin:$HOME/.juliaup/bin:$PATH"

julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.test()'
JULIA_NUM_THREADS=1 julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

julia --project=Mimosa.jl -e \
  'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'

julia --project=Mimosa.jl/docs Mimosa.jl/docs/make.jl
julia --project=Mimosa.jl/test Mimosa.jl/test/downstream/runtests.jl
```

После добавления JET и отдельных environments команды уточняются в README/CI и не должны зависеть от stale local
Manifest files. Benchmark suite запускается отдельно и не входит в обычный correctness test job.

## 14. Gate checklist

### Gate R1 — Correctness restored

- [ ] motif/profile null strategy выполняет правильный algorithm path;
- [ ] CLI передаёт все принятые options;
- [ ] summary, manifest и фактический result согласованы;
- [ ] compatibility и serial/threaded tests проходят;
- [ ] отсутствуют `Any` и string dispatch в null build configuration path.

### Gate R2 — Untrusted input hardened

- [x] bundle path traversal и symlink escape запрещены;
- [x] checksum обязателен и строго валидируется;
- [x] NPY schema и size limits проверяются до allocation;
- [x] encoded bases и destination sizes валидируются до `@inbounds`;
- [x] hostile corpus возвращает controlled typed errors для bundle boundary.

### Gate R3 — CI and contracts enforced

- [ ] Julia matrix работает на minimum/stable и serial/threaded configurations;
- [ ] Aqua, JET, formatter, docs/doctests и precompile fail closed;
- [ ] subprocess CLI и security jobs включены;
- [ ] downstream consumer использует отдельный clean environment;
- [ ] fresh install проходит без Python.

### Gate R4 — Documentation and evidence complete

- [ ] feature matrix отражает все user-facing capabilities;
- [ ] README, quick start и CLI examples исполняются;
- [ ] каждый exported symbol документирован;
- [ ] latency/performance report опубликован с полной environment metadata;
- [ ] stable benchmark baseline утверждён.

### Gate R5 — Release candidate ready

- [ ] Stage 10 platform matrix пройдена;
- [ ] registry, license, citation и release notes проверены;
- [ ] нет high/critical undocumented compatibility differences;
- [ ] downstream acceptance получен;
- [ ] migration window и rollback policy опубликованы.

## 15. Правила выполнения и отчётности

- Один merge request закрывает один логический risk area; scientific behavior, storage schema и performance refactor не
  смешиваются без необходимости.
- Сначала добавляется regression/security fixture, затем исправление, затем полный relevant test run.
- Frozen Python fixtures не регенерируются без отдельного review научного изменения.
- Не удалять пользовательские или параллельные изменения из dirty working tree.
- Не добавлять `@inbounds`, SIMD, generated functions, fast-math или новые dependencies без profile evidence.
- Каждый закрытый пункт обновляет этот checklist, `PLAN.md`, feature matrix и при необходимости `AGENTS.md` в одной
  change set, чтобы статусы больше не расходились.
- Отчёт содержит exact commands, versions, compatibility result, benchmark context, known differences и остаточные
  риски.

## 16. Definition of Done для remediation plan

`PLAN_2.md` считается выполненным только после прохождения Gate R1–R5. Большое число unit tests само по себе не
является основанием закрыть gate. Нужны доказательства корректного option wiring, hostile-input safety, clean CI,
исполняемой документации, воспроизводимых performance measurements и установки release artifact без Python runtime.
