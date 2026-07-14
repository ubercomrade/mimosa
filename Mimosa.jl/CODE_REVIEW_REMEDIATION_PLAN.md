# План устранения дефектов по результатам code review Mimosa.jl

Статус: в реализации; отмеченные ниже пункты отражают проверенное состояние
этого worktree, а не исходную оценку code review.

Дата ревизии: 2026-07-13.

Связанный документ: `ARCHITECTURE_REFACTORING_PLAN.md`. Этот план дополняет
архитектурный план конкретными дефектами безопасности, корректности, хранения,
публичного API и тестового покрытия. Он не разрешает менять численные алгоритмы,
форматы или frozen fixtures без отдельного решения о совместимости.

## 1. Цели

- Устранить возможность чтения, записи и удаления файлов вне cache root.
- Сделать все публичные границы безопасными до входа в `@inbounds`-код.
- Обеспечить однозначную совместимость null distributions с параметрами
  сравнения и исходными данными.
- Привести фактическую атомарность model/null/cache writes в соответствие с
  документацией.
- Сделать parsers действительно ограниченными по памяти и возвращающими
  предсказуемые typed errors.
- Разделить scannable motifs, precomputed profiles и промежуточные типы по их
  возможностям, не полагаясь на наличие неформальных полей.
- Сократить случайную публичную поверхность, мертвый код и дублирование без
  нарушения downstream contract.

## 2. Ограничения

- Не изменять порядок Float32/Float64 вычислений, редукций и tie-breaking без
  compatibility evidence.
- Не ослаблять проверки bundle paths, checksums, NPY headers и allocation
  budgets.
- Не восстанавливать удаленный root Python package или root oracle fixtures.
- Не регенерировать frozen fixtures для прохождения тестов.
- Не совмещать schema migration, перемещение файлов и изменение научного
  алгоритма в одном change set.
- Любое изменение model/null/cache schema требует обновления format version,
  migration policy, документации и hostile-input tests.
- Сохранять Julia 1.10 compatibility и deterministic serial/threaded output.

## 3. Приоритеты

| Приоритет | Область | Риск |
|---|---|---|
| P0 | Cache containment и destructive clear | Запись/чтение/удаление вне cache root |
| P0 | Публичные `@inbounds` boundaries | Небезопасный доступ к памяти |
| P1 | Null compatibility и identifiers | Некорректные p-value/E-value и provenance collisions |
| P1 | ScoreProfile fingerprint и duplicate names | Коллизии cache/null metadata и неверные пары |
| P1 | Atomic bundle/cache writes | Потеря ранее валидного артефакта при сбое |
| P1 | PreparedProfile threshold compatibility | Сравнение несовместимых anchors |
| P1 | Bounded scientific parsers | Неконтролируемые аллокации и malformed profiles |
| P2 | Model capability hierarchy и Float32 contract | MethodError и разные численные типы public paths |
| P2 | GEV/statistics validation | Несошедшиеся fit objects и некорректные вероятности |
| P2 | CLI/API organization | Нестабильные exit codes и случайные exports |
| P3 | Dead code и allocation cleanup | Сложность сопровождения и лишние аллокации |

## 4. Этап 0. Восстановить проверяемый baseline

Цель этапа: получить возможность запускать независимые unit/security tests, не
восстанавливая удаленный root fixture corpus.

- [x] Принять отдельное решение о владельце compatibility fixtures, на которые
  ссылается `test/runtests.jl`.
- [x] Не читать root `tests/fixtures/compatibility/manifest.json` до unit tests,
  которым fixture metadata не требуется.
- [x] Либо перенести согласованный frozen corpus в package-local fixtures с
  зафиксированным provenance, либо отделить fixture suite от unit suite.
  Принято второе решение: corpus не восстанавливается и compatibility suite
  получает явный broken/skip status при его отсутствии.
- [x] Не заменять отсутствие fixtures безусловным `skip`: CI должен явно
  сообщать, какой compatibility contract не был проверен.
- [x] Зафиксировать исходный список exports, Julia version, runtime threads и
  focused benchmark baseline до любых численных изменений.

Критерий готовности: unit, security, JET и Aqua suites могут запускаться без
root Python fixtures; fixture suite либо проходит на утвержденном corpus, либо
дает отдельный явный статус.

## 5. Этап 1. Закрыть cache и memory-safety дефекты

### 5.1. Cache path containment

- [x] Добавить единую `_validate_cache_key` и применять ее в `cache_has`,
  `cache_get`, `cache_get_meta`, `cache_set` и обоих `clearcache`.
- [x] Разрешать только один безопасный path component. Запретить абсолютные
  пути, `/`, `\\`, NUL, `.`/`..` и platform drive prefixes.
- [x] После построения пути проверять нормализованный/real path containment под
  cache root; не полагаться только на regex.
- [x] Решить, является ли публичным контрактом произвольный человекочитаемый
  key. Предпочтительно ввести внутренний `CacheKey` либо ограниченный ASCII
  key, сохранив текущие тестовые имена без traversal components.
- [x] `clearcache(cache)` должен удалять только распознанные Mimosa entries и
  оставлять unrelated files и directories без изменений.
- [x] CLI `cache clear` должен отклонять подозрительный cache root и сообщать
  число удаленных entries, а не число произвольных файлов.

Обязательные тесты: `../`, абсолютный путь, backslash, symlink escape, NUL,
unrelated sentinel file, nested directory, disabled cache и single-entry clear.

### 5.2. Безопасные публичные builders

- [x] В `build_anchor_csr` до `@inbounds` проверить `n_rows >= 0`, равенство
  длин `rows`/`positions`, `1 <= row <= n_rows` и положительность positions.
- [ ] Добавить внутренний `_build_anchor_csr_unchecked` только если benchmark
  показывает стоимость повторной проверки на уже проверенном hot path.
- [x] Сделать `AnchorCSR` валидируемым типом: offsets начинаются с 1,
  не убывают, заканчиваются `length(positions)+1` и имеют ожидаемое число rows.
- [x] В `build_pcm` проверить положительный `motif_width`, точное соответствие
  `size(sites, 1)` и допустимые DNA codes до `@inbounds`.
- [x] В `extract_site_matrix` проверить motif width, sequence indices, starts,
  strands и переполнение арифметики `start + width - 1` до выделения/копирования.
- [x] Добавить constructor validation для `SiteCollection`, не ограничиваясь
  равенством длин параллельных массивов.

Обязательные тесты: mismatched arrays, row 0, row `n_rows+1`, negative sizes,
malformed CSR offsets, invalid strand/code, overflow-oriented large integers и
проверка, что ошибки являются `ArgumentError`/`InvariantError`, а не BoundsError.

## 6. Этап 2. Исправить storage atomicity и fingerprints

### 6.1. Bundle commit protocol

- [x] Не считать `mv(stage, target; force=true)` атомарной заменой непустого
  каталога: Julia может удалить существующий target перед вторым rename.
- [x] Рекомендуемый безопасный default: создавать только новый target и явно
  отказывать, если он существует.
- [x] Если overwrite необходим, ввести отдельный opt-in recovery protocol с
  sibling backup, fsync родительского каталога и восстановлением после сбоя.
  Overwrite не является публичным контрактом: текущий writer безопасно
  отказывает при существующем bundle; cache replacement использует backup.
- [ ] Fault-injection tests должны прерывать writer до manifest, после blobs и
  непосредственно перед commit; существующий валидный bundle обязан остаться
  читаемым.
- [x] Заменить `writer::Function` на параметрический callable `where {F}`.

### 6.2. Cache entry format

- [x] Не коммитить data и metadata двумя независимыми rename под общими `.tmp`
  именами.
- [x] Предпочтительно хранить entry как отдельный staged directory и коммитить
  directory целиком. Это требует осознанного `CACHE_FORMAT_VERSION` bump.
- [x] Использовать уникальные sibling stages для concurrent writers.
- [x] Запретить user metadata переопределять `format_version`, `checksum` и
  `size`.
- [x] Либо выполнить настоящий fsync temp file и parent directory, либо убрать
  вводящее в заблуждение имя `_fsync_and_rename` и обещание durability.
- [x] Добавить recovery/cleanup policy для orphan stages без удаления валидной
  entry.

### 6.3. Content fingerprints

- [x] Реализовать отдельный `content_fingerprint(::ScoreProfile)`, включающий
  kind, name, score data, offsets, dtype и layout marker.
- [x] Общий fallback для неизвестного `AbstractMotifModel` должен бросать
  явную ошибку, а не хешировать только type/name.
- [x] Канонизировать integer/float byte order и integer width, если fingerprints
  заявлены переносимыми между архитектурами.
- [x] Зафиксировать, должна ли конкретная matrix wrapper type влиять на
  fingerprint семантически одинаковой модели.
- [x] Добавить collision tests для одинаковых имен и разных score data/offsets.

## 7. Этап 3. Зафиксировать null compatibility contract

Этот этап меняет storage schema и требует ADR или обновления storage format
specification. Текущий `NULL_FORMAT_VERSION = 2` нельзя менять неявно.

- [x] Ввести типизированное описание profile comparison contract: metric,
  `search_range`, `window_radius`, `realign_window`, `min_logfpr`, sequence и
  background fingerprints, а также версии normalization/alignment algorithms.
- [x] Сохранять этот contract в null manifest и проверять его в библиотечном
  API до annotation. CLI должен только преобразовывать ошибку в exit code.
  Для library annotation проверяется валидность/fit contract; сравнение с
  внешними input fingerprints пока выполняется CLI boundary.
- [x] При изменении manifest выпустить следующий null format version и явно
  решить: reject v2, read-only migration либо ограниченная совместимость.
- [x] Строить `null_id` из canonical compatibility metadata, checksum raw scores
  и fit metadata. Использовать `NULL_FORMAT_VERSION`, а не hardcoded version.
- [x] Проверять `strategy == "profile"` и поддерживаемый metric уже в
  `savenull`, а не только при последующей загрузке.
- [x] Решить, нужно ли сохранять `NullPair` и исходное сообщение
  `GEVFitFailure`; сейчас round-trip теряет эти данные.
- [x] Отклонять duplicate/empty model names до подготовки профилей и построения
  `Dict` по именам.
- [x] Проверять уникальность relation names и однозначность group assignment.

Обязательные тесты: каждый измененный alignment parameter делает bundle
несовместимым; разные sequences/background дают разные `null_id`; v2 migration
поведение; duplicate models; serial/threaded equality всех сохраненных полей.

## 8. Этап 4. Исправить PreparedProfile и модельные возможности

- [x] Scalar `compare(PreparedProfile, model, sequences)` и обратный overload
  должны по умолчанию наследовать threshold подготовленной стороны.
- [x] Явно переданный threshold обязан точно совпадать с `min_logfpr` prepared
  profile; использовать тот же helper, что one-to-many paths.
- [x] Валидировать row counts, bundle offsets и anchor row counts перед
  alignment.
- [x] Убрать или валидировать публичный трехаргументный конструктор
  `PreparedProfile`, позволяющий создать несовместимые bundle/anchors.
- [x] Разделить `AbstractMotifModel` на возможности или ввести четкий dispatch:
  scannable motif, matrix requiring PWM conversion и precomputed profile.
- [x] Определить поддерживаемую семантику mixed `ScoreProfile`/motif comparison.
  Либо реализовать ее через общую normalization pipeline, либо отклонять на
  parse/API boundary и убрать комбинацию из CLI help.
- [x] Явно определить поддержку PFM: автоматическая конверсия в PWM или
  документированный запрет прямого scan/compare.
- [x] Устранить неоднозначную семантику `length(ScoreProfile)` против motif
  length; generic code должен использовать именованные accessors.
- [x] Higher-order generic methods должны использовать `scorematrix`,
  `motif_length`, `window_size` и geometry accessors, а не предполагать поля
  `representation`, `span` и `motif_length`.
- [x] Решение о каноническом Float32 scan output принять в рамках раздела 3.1
  `ARCHITECTURE_REFACTORING_PLAN.md`, затем закрепить constructor/API tests.

## 9. Этап 5. Усилить parsers и relation input

- [ ] Ввести общие limits для file bytes, line bytes, row count и total numeric
  elements для score, relation, MEME/PFM, BaMM и SiteGA inputs.
  Частично реализовано для score/relation/PFM/SiteGA/XML; BaMM/Slim и общий
  shared limits layer ещё требуют отдельной работы.
- [x] Проверять размер файла до `read`, `readlines` или полной materialization;
  затем читать streaming там, где формат это допускает.
- [x] XML format detection должен читать ограниченный prefix либо использовать
  уже разобранный bounded DOM, не читать файл второй раз целиком.
- [x] `read_scores` должен требовать корректную header/row структуру, сохранять
  пустые rows в точном порядке, отклонять пустой файл и non-finite scores.
- [ ] Все parser errors, включая missing files и malformed Unicode/XML, должны
  нормализоваться в `ModelFormatError` с path/context.
- [x] Relation parser не должен удалять внутренние пустые headers с изменением
  индексов колонок.
- [ ] Явно поддержать quoted CSV либо документировать и проверять только
  простой delimiter-separated формат.
- [x] Отклонять duplicate headers, слишком длинные/короткие rows, пустые group
  names и противоречивые duplicate motif assignments.

Обязательные тесты: oversized input до allocation, empty profiles, consecutive
headers, NaN/Inf, quoted delimiter policy, empty middle header, duplicate motif,
malformed UTF-8/XML и точный тип исключения.

## 10. Этап 6. Исправить statistics и site-selection validation

- [x] `fit_gev` должен проверять `max_iter > 0`, конечный `tol > 0` и возвращать
  `GEVFitFailure` при несходимости согласно docstring либо изменить публичный
  контракт и всех callers.
- [x] Валидировать `GEVFit`: finite parameters, positive scale, non-negative
  iterations и finite loglikelihood.
- [x] Добавить `_numerical_gradient!` и переиспользовать buffer вместо замены
  заранее выделенного `g_new` на каждой BFGS iteration.
- [x] `adjusted_pvalues` должен отклонять non-finite значения и значения вне
  `[0, 1]`.
- [x] `evalue` и `annotate_results` должны проверять неотрицательное effective
  number of targets и использовать один helper для вычисления.
- [x] Упростить `annotate_results`: `valid_indices` и fallback через
  `isassigned` недостижимы при текущем цикле.
- [x] Валидировать `TopFractionHits` как конечную fraction в согласованном
  диапазоне, предпочтительно `0 < fraction <= 1`.
- [x] Сделать `TopFractionHits{S<:SiteSelector}` параметрическим вместо поля
  `base::SiteSelector`.

Обязательные тесты: negative/NaN probabilities, invalid effective count,
non-converged GEV, invalid scale, fraction boundaries и отсутствие новых
allocations в benchmark GEV fit.

## 11. Этап 7. Упорядочить CLI, exports и naming

- [x] Оставить единственный список `export` в `src/Mimosa.jl`; include-файлы не
  должны менять публичную поверхность.
- [ ] Сверить список с downstream contract и ввести deprecation cycle для
  удаляемых public aliases.
- [x] Убрать `_fit_transform_empirical` из exports либо переименовать и
  документировать как стабильный public API.
- [ ] Унифицировать порядок аргументов `npositions`; model-specific aliases
  удалять только через deprecation policy.
- [ ] Не проводить массовое переименование `readmodel`/`scorebounds` вместе с
  функциональными исправлениями. Сначала выбрать naming policy и compatibility
  aliases.
- [x] Перенести null compatibility validation из CLI в library API.
- [x] Добавить typed CLI parse helpers для Int/Float32/ranges, чтобы invalid
  arguments стабильно давали exit code 1.
- [x] Исправить global `--quiet`/`--verbose` перед command и проверять точное
  число positional arguments.
- [x] Удалить или реализовать `build-null --cache-dir`, аргумент `pattern` и
  неиспользуемый parser parameter.
- [ ] Объединить JSON writer CLI с `serialization.jl`, сохранив stdout/stderr и
  schema contract.
- [x] Получать CLI/provenance version из package version, не дублировать
  строку `0.1.0` в нескольких файлах.

## 12. Этап 8. Удалить мертвый код и очевидные аллокации

Удалять только после повторного `rg`, downstream audit и проверки, что имя не
является документированным extension point.

- [x] Удалить `_metric_string`, если для него не найден caller.
- [x] Удалить неиспользуемый `nparams`; `lookup_score_for_tail_probability`
  оставлен как внутренний документированный helper для normalization API.
- [x] Удалить неиспользуемые XML wrappers `xml_tag` и `xml_findall` либо
  использовать их последовательно вместо прямого доступа к полям.
- [x] Удалить недостижимую проверку `fraction === nothing` при аргументе
  `Float64`.
- [x] Удалить неиспользуемый keyword `padding` из `from_padded` либо придать ему
  проверяемую семантику.
- [x] Валидировать/реализовать `writemodel(...; format)` либо удалить keyword с
  deprecation period.
- [x] Строить `make_random_sequences`, batch reverse complement и
  `from_padded` сразу в flat buffers без `Vector{Vector}` staging.
- [ ] После разделения ответственности разбить `cli.jl`, `alignment.jl` и
  `sites.jl` по назначению, не создавая отдельный Julia module на каждый файл.

## 13. Расширение тестового контроля

- [ ] Расширить JET с одного prepared-to-prepared smoke test на scanning,
  prepared/model scalar и one-to-many, cache key boundary, null build,
  annotation и GEV public paths.
- [x] Включить Aqua ambiguity и unbound-args checks, если выявленные результаты
  приняты как baseline; не держать их постоянно выключенными без причины.
- [ ] Добавить security corpus для cache traversal/destructive clear и parser
  allocation limits.
- [ ] Добавить crash/fault-injection tests для model/null/cache commits.
- [ ] Проверять одинаковые serial/threaded order, discrete fields, exceptions и
  absence of partially usable outputs.
- [ ] Для fingerprint/null identifiers добавить golden canonicalization tests
  на Julia 1.10 и latest stable.
- [ ] После каждого этапа запускать narrow tests, затем downstream contract,
  formatter, docs и доступную часть full suite.

Рекомендуемые команды:

```bash
julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_validation.jl")'

julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_cache.jl")'

JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_parallel.jl")'

julia --project=Mimosa.jl/test/downstream Mimosa.jl/test/downstream/runtests.jl

julia --project=Mimosa.jl/test -e \
  'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'
```

## 14. Рекомендуемое разбиение change sets

1. Cache path validation и non-destructive clear без schema change.
2. Anchor/site public validation и hostile-input tests.
3. ScoreProfile fingerprint, duplicate-name rejection и tests.
4. PreparedProfile threshold compatibility и mixed-type API decision.
5. Bundle overwrite protocol и fault-injection tests.
6. Cache entry schema/atomicity с version bump и migration policy.
7. Null compatibility schema с version bump и migration policy.
8. Bounded score/relation/XML detection parsers.
9. GEV, p-value и site-selector validation.
10. CLI typed parsing, global flags и library-first compatibility checks.
11. Export consolidation, deprecations и dead-code cleanup.
12. File reorganization и performance cleanup после повторного benchmark.

## 15. Definition of Done

### Текущий аудит реализации

- Cache format поднят до 2: entry коммитится staged directory, stages имеют
  уникальные имена, замена существующей entry выполняется через sibling backup
  с восстановлением; durability через `fsync` намеренно не обещается.
- Null format поднят до 3: manifest хранит comparison contract, raw-score
  fingerprint и `NullPair`; v2 намеренно reject-only, без неявной миграции.
- Compatibility corpus не восстанавливается: unit/security/CLI suite отделён,
  а отсутствие corpus имеет явный broken status в full suite.
- Не закрыты без отдельного benchmark/fault-injection change set: flat-buffer
  refactor для всех parser/model staging paths, crash injection matrix для
  bundle/cache commits и расширенный JET workload beyond текущего smoke path.
  Эти пункты оставлены unchecked выше намеренно; их компенсация не меняет
  численные контракты и не маскирует отсутствие проверки.

- Ни одна публичная функция не входит в `@inbounds` region до проверки всех
  зависимых индексов, размеров и codes.
- Cache operations не могут выйти за root и не удаляют unrelated files.
- Сбой записи не заменяет и не уничтожает ранее валидный model/null/cache
  artifact.
- Null annotation проверяет полный численный и data compatibility contract.
- Семантически разные ScoreProfile/model inputs имеют разные fingerprints.
- Все parsers имеют pre-allocation bounds и возвращают typed contextual errors.
- Public scan result dtype определен, документирован и одинаков для всех model
  families.
- Export list единственный, проверен downstream suite и не содержит случайных
  underscored helpers.
- Мертвый код удален только после downstream/documentation audit.
- Narrow tests, Aqua, expanded JET, downstream contract, formatter и docs
  проходят; full-suite fixture status сообщен отдельно и точно.
- Численные fixtures, tie-breaking, coordinates, schema versions и parallel
  ordering не изменены без явного compatibility decision.
