# План исправления проблем Mimosa

## Цель

Устранить найденные ошибки корректности и безопасности, затем сократить время и
пиковую память основных путей `prepare_profile`, `compare_many`, `build_null` и
работы с дисковым кэшем. Изменения должны сохранять порядок результатов,
JSON-only `stdout` CLI и воспроизводимость статистических расчётов.

## Правила выполнения

- Сначала исправлять ошибки, меняющие научный результат или допускающие потерю
  данных; оптимизации не должны предшествовать этим исправлениям.
- Для каждого нетривиального исправления добавлять минимальный regression-тест,
  который падает на текущем коде.
- Не восстанавливать небезопасную загрузку legacy pickle. Удалить ложное
  обещание совместимости и соответствующий ложноположительный тест.
- Не добавлять новые зависимости без измеримой необходимости.
- После изменения форматов кэша или статистических контрактов повышать их
  версии и проверять поведение на несовместимых данных.
- Выполнять этапы последовательно; P0 и P1 должны быть завершены до оптимизаций.

## Этап 0. Зафиксировать исходное состояние

- [ ] Запустить `uv sync --locked --group dev`.
- [ ] Запустить `uv run ruff check .`.
- [ ] Запустить `uv run pytest -q`.
- [ ] Запустить `uv run bash examples/run.sh`.
- [ ] Запустить минимальный production benchmark и сохранить результат вне
  репозитория, например в `/tmp/mimosa-performance-before.json`.
- [ ] Зафиксировать время, размер кэша и параметры CPU/потоков для повторного
  сравнения после этапов производительности.

Критерий завершения: известен baseline корректности и производительности; все
исходные падения отделены от новых regression-тестов.

## Этап 1. P0: безопасность и корректность результатов

### 1.1 Безопасный `cache clear`

Файлы: `src/mimosa/cache.py`, `src/mimosa/cli.py`, `tests/test_cache.py`,
`tests/test_cli.py`.

- [ ] Удалить wildcard-удаление каталогов, содержащих `.backup-`.
- [ ] Не удалять staging-каталоги только по имени без подтверждения, что ими
  владеет Mimosa; безопасный минимум — очищать только валидные cache entries.
- [ ] Сохранить запрет на `/`, домашний каталог, файлы и symlink-каталоги.
- [ ] Добавить тесты, что каталоги `user.backup-data` и
  `.mimosa-cache-stage-user-data` с пользовательскими файлами сохраняются.

Критерий завершения: `cache clear` удаляет только записи с валидной cache
metadata и никогда не удаляет посторонний каталог по совпадению имени.

### 1.2 Исправить empirical upper-tail lookup

Файлы: `src/mimosa/_kernels.py`, `tests/test_profiles.py`.

- [ ] Для descending table выбирать последнюю калибровочную оценку `>= target`,
  корректно ограничивая значения ниже минимума и выше максимума.
- [ ] Применить одинаковую семантику к exact empirical и exact tail hybrid
  normalization.
- [ ] Исправить тест, который сейчас закрепляет неверный результат для `2.5`
  между `3.0` и `2.0`.
- [ ] Добавить тест с отдельным background, где foreground score отсутствует в
  калибровочной выборке.
- [ ] Обновить ожидаемые reference scores, только если изменение объясняется
  исправленной формулой.

Критерий завершения: normalized score соответствует
`-log10(count(calibration >= score) / n)` для значений на сетке, между точками и
за границами таблицы.

### 1.3 Выравнивать модели по физическим координатам сайтов

Файлы: `src/mimosa/profiles/prepared.py`,
`src/mimosa/profiles/alignment.py`, `src/mimosa/compare.py`,
`src/mimosa/cache.py`, `src/mimosa/models.py`, `tests/test_profiles.py`.

- [ ] Добавить в prepared profile минимальную координатную информацию:
  `site_start_offset` исходной модели; для `ScoreProfile` использовать `0`.
- [ ] Учитывать разницу offsets query/target при alignment и при формировании
  публичного `ComparisonResult.offset`.
- [ ] Не смешивать scan index и физическую координату сайта в anchors.
- [ ] Включить координатный offset в сериализацию и cache key; повысить версию
  prepared-profile cache/alignment contract.
- [ ] Добавить тест эквивалентных PWM и order-1 BaMM: физический offset должен
  быть `0`, включая `search_range=0`.
- [ ] Проверить обе ориентации и модели с разными `left_context`.

Критерий завершения: одинаковые физические сайты сравниваются в одной
координате независимо от контекста модели.

### 1.4 Защитить инварианты массивов перед Numba

Файлы: `src/mimosa/arrays.py`, `src/mimosa/profiles/anchors.py`,
`src/mimosa/profiles/prepared.py`, `src/mimosa/scan.py`,
`tests/test_arrays.py`, `tests/test_profiles.py`, `tests/test_scan.py`.

- [ ] `EncodedSequences` должен владеть валидированными `data` и `offsets` и
  предоставлять их read-only; заменить тест, требующий alias/mutation.
- [ ] Свежие normalized scores, offsets и anchors prepared profile сделать
  read-only после построения.
- [ ] В `PreparedProfile` проверить finite Float32 scores, одинаковые row
  layouts forward/reverse, supported normalization и Float32-representable
  `min_logerr`.
- [ ] Для custom `scan_into` инициализировать output NaN и отклонять
  незаполненные, NaN или Inf значения.
- [ ] Проверить finite output встроенных scanners после возможного Float32
  overflow.

Критерий завершения: после успешной валидации публичная мутация не может
передать неверный DNA code, offset или anchor в bounds-unchecked Numba kernel.

### 1.5 Закрыть выход bundle за пределы root

Файлы: `src/mimosa/io/bundles.py`, `tests/test_io.py`.

- [ ] Разрешать root и candidate через `realpath`/`Path.resolve`.
- [ ] Проверять containment через `os.path.commonpath`.
- [ ] Отклонять symlink в любом компоненте пути, а не только конечный файл.
- [ ] Добавить тест с `bundle/data` как symlink на внешний каталог.

Критерий завершения: manifest не может заставить reader открыть файл вне
реального bundle root.

## Этап 2. P1: публичные контракты и численная корректность

### 2.1 Согласовать prepared API

Файлы: `src/mimosa/compare.py`, `tests/test_profiles.py`.

- [ ] В `compare` отклонять явно переданные `min_logerr` и `normalization`, если
  они отличаются от prepared profile, как уже делает `compare_many`.
- [ ] Проверять совместимость обеих prepared сторон до alignment.
- [ ] Всегда включать `n_sites`, включая `0`, в `ComparisonResult.to_dict()`.

### 2.2 Исправить смысл `n_sites`

Файлы: `src/mimosa/_kernels.py`, `src/mimosa/profiles/alignment.py`,
`tests/test_profiles.py`.

- [ ] Считать site contributing только при finite metric contribution.
- [ ] Использовать contributing count, а не attempted count, в tie-breaking.
- [ ] Добавить constant/zero-norm profile tests для всех metrics.

### 2.3 Строго проверять целочисленную геометрию

Файлы: `src/mimosa/profiles/alignment.py`, `src/mimosa/statistics.py`,
`src/mimosa/models.py`, `src/mimosa/io/models.py`, тесты соответствующих
модулей.

- [ ] Отклонять bool, float и string для search/window/realign ranges.
- [ ] Ввести один небольшой strict integer validator для model geometry и
  parser fields.
- [ ] Не преобразовывать `order=1.9`, XML `length=15.9` или `True` через
  безусловный `int()`.
- [ ] Хранить в null contract уже проверенные integer values.

### 2.4 Усилить валидацию моделей и матриц

Файлы: `src/mimosa/io/models.py`, `src/mimosa/models.py`,
`src/mimosa/sites.py`, `tests/test_io.py`, `tests/test_models.py`,
`tests/test_sites.py`.

- [ ] Проверять BaMM probability groups: finite, диапазон `[0, 1]`, сумма
  приблизительно `1.0` для каждой conditional группы.
- [ ] Проверять `ndim == 2` до обращения к `shape[0]`/`shape[1]` в matrix
  helpers.
- [ ] В `build_pcm` отклонять DNA codes вне `0..4`, включая отрицательные.
- [ ] В `ThresholdHits` требовать конечный Float32 threshold.
- [ ] В `pcm_to_pfm` выполнять overflow-safe вычисление или отклонять
  pseudocount, не представимый безопасно; проверить finite output и суммы
  колонок.

### 2.5 Сделать statistical objects стабильными

Файлы: `src/mimosa/statistics.py`, `src/mimosa/io/bundles.py`,
`tests/test_statistics.py`, `tests/test_io.py`.

- [ ] В `NullDistribution.__post_init__` копировать и замораживать raw scores,
  пары и contract либо хранить их в неизменяемом представлении.
- [ ] Проверять rank, finite scores, обязательные contract fields и fingerprint.
- [ ] Не допускать одинаковый `null_id` после изменения raw scores.
- [ ] Сохранять в pairs различие original/shuffled через стабильные variant IDs,
  а не только одинаковые model names.

### 2.6 Довести `SiteCollection` до заявленного контракта

Файлы: `src/mimosa/sites.py`, `docs/python/api.md`, `tests/test_sites.py`.

- [ ] Преобразовывать входы в одномерные массивы нужных dtypes и отклонять
  дробные индексы до cast.
- [ ] Заморозить внутренние массивы либо явно документировать mutability.
- [ ] Реализовать небольшой `to_dict()` либо удалить его из документации;
  предпочтительно реализовать, так как метод уже заявлен публично.

## Этап 3. P1: быстрые оптимизации с очевидным эффектом

### 3.1 Сортировать null distribution один раз

Файлы: `src/mimosa/statistics.py`, `tests/test_statistics.py`.

- [ ] В `annotate_results` один раз валидировать и сортировать raw scores.
- [ ] Для каждого result выполнять только `np.searchsorted`.
- [ ] Добавить тест, подтверждающий один вызов sort на batch annotation.

Ожидаемая сложность:
`O(N log N + targets log N)` вместо `O(targets * N log N)`.

### 3.2 Не пересчитывать одинаковые null pairs

Файлы: `src/mimosa/statistics.py`, `tests/test_statistics.py`.

- [ ] Memoize deterministic comparison по `(query_idx, target_idx)`.
- [ ] Сохранить исходные sampled work items, порядок raw scores и pairs.
- [ ] Добавить тест, что число `compare` calls не превышает число уникальных
  sampled pairs.

Для трёх моделей максимум допустимых пар равен 24 при default `n_samples=2000`,
поэтому это потенциально крупнейшее сокращение CPU в `build_null`.

### 3.3 Не удерживать полный sparse anchor buffer

Файлы: `src/mimosa/profiles/anchors.py`, `tests/test_profiles.py`.

- [ ] После заполнения копировать `positions[:count]`, если capacity заметно
  больше count.
- [ ] Если peak allocation остаётся проблемой, заменить на count pass + exact
  allocation + fill pass.
- [ ] Добавить тест на фактический `positions.nbytes`, а не только `.size`.

### 3.4 Убрать очевидные полноразмерные temporaries

Файлы: `src/mimosa/scan.py`, `src/mimosa/profiles/normalization.py`, тесты.

- [ ] Не выделять `data` для `strands="both"`.
- [ ] Для `best` использовать `np.maximum(fwd, rev, out=data)` вместо
  `np.where` с временным result.
- [ ] Не копировать shared calibration strand без необходимости.
- [ ] Переиспользовать immutable offsets вместо `.copy()`.
- [ ] Освобождать raw/calibration references до cache serialization.

### 3.5 Соблюдать thread budget Python API

Файлы: `src/mimosa/compare.py`, `tests/test_profiles.py`.

- [ ] Временно устанавливать `inner_threads` для parent query preparation и
  serial target pipeline.
- [ ] Восстанавливать предыдущий Numba thread count после вызова.
- [ ] Ограничивать joblib workers числом targets.
- [ ] Добавить тесты thread count внутри query preparation и serial path.

## Этап 4. P2: оптимизации после измерений

### 4.1 Fingerprints и cache context

Файлы: `src/mimosa/io/bundles.py`, `src/mimosa/cache.py`,
`src/mimosa/compare.py`, `src/mimosa/statistics.py`.

- [ ] Хэшировать contiguous bytes через `memoryview`, без `bytes(data)`.
- [ ] Переиспользовать один sequence/background preparation context в
  `compare` и `build_null`, как уже делает `compare_many`.
- [ ] Измерить Python textual bit fingerprinting; оптимизировать чанками только
  при подтверждённом bottleneck, сохранив Julia-compatible digest.
- [ ] Убрать повторное checksum-чтение model bundle payload.

### 4.2 Cache write и mmap hit

Файлы: `src/mimosa/cache.py`, `tests/test_cache.py`, benchmark.

- [ ] Измерить encode, lock wait, write, checksum и semantic validation
  отдельно.
- [ ] При подтверждённом memory peak писать sections сразу в staged file с
  incremental SHA-256 вместо сборки полного `bytearray` и `bytes`.
- [ ] Переходить на per-key locks только при измеренном write contention.
- [ ] Рассмотреть один mmap payload с section views вместо нескольких mappings.
- [ ] Закрыть риск stale `_verified_entries`: equal-size mutation после первого
  hit не должна проходить без повторной проверки.

### 4.3 Alignment memory и serial scratch

Файлы: `src/mimosa/_kernels.py`, `src/mimosa/profiles/alignment.py`, benchmark.

- [ ] Заменить `seen.fill(0)` на epoch strategy в serial CSR path.
- [ ] Измерить память `24 * rows * shifts` на worker.
- [ ] При подтверждённом bottleneck заменить row matrices на thread-local
  reductions, сохранив детерминированный порядок суммирования.
- [ ] Подбирать parallel dispatch по rows, shifts, windows и anchor density, а
  не только по общему числу score items.

### 4.4 Parsing больших входов

Файлы: `src/mimosa/io/fasta.py`, `src/mimosa/io/models.py`, `src/mimosa/cli.py`.

- [ ] Добавить общий лимит FASTA bases/bytes, а не только лимиты строки и одной
  последовательности.
- [ ] Парсить FASTA в один растущий buffer плюс offsets, без списка полных
  sequence copies.
- [ ] Парсить score lines числовыми chunks (`np.fromstring`) вместо Python float
  object на каждый элемент.
- [ ] Ограничить размер XML до parsing и глубину Dimont tree.
- [ ] Не использовать `readlines()` для SiteGA.

### 4.5 Дубликаты targets и higher-order scratch

- [ ] Измерить частоту идентичных targets и joblib serialization overhead.
- [ ] При реальной пользе deduplicate только безопасные immutable/prepared
  targets по identity и восстановить исходный порядок результатов.
- [ ] Проверить замену Int64 rolling codes на Int32 при максимальном
  поддерживаемом order.

## Этап 5. Исправить benchmark

Файл: `benchmarks/benchmark_performance.py`, `tests/test_benchmark.py`.

- [ ] Запускать каждый memory sample в отдельном subprocess.
- [ ] Измерять aggregate process-tree RSS, включая joblib workers.
- [ ] Если это не реализовано, переименовать поле в
  `parent_lifetime_max_rss_bytes` и не использовать его для сравнений памяти.
- [ ] Разделить и явно назвать `cache_miss_hot_jit`,
  `cache_hit_hot_filesystem`, `prepared` и cold-JIT CLI modes.
- [ ] Добавить phase timings: scan, normalize, anchors, fingerprint, cache
  encode/write/read, alignment и serialization.
- [ ] Добавить BaMM и SiteGA, higher-order model, skewed row lengths, большой
  background и threshold density (`0`, epsilon, `1`, `2`, `4`).
- [ ] Проверить dispatch boundaries: 63/64 rows и 49 999/50 000 items.
- [ ] Использовать `tempfile.gettempdir()` вместо hard-coded `/tmp`; gracefully
  отключать `resource`-метрики на неподдерживаемых платформах.

Критерий завершения: benchmark измеряет именно production paths и позволяет
сравнить время и aggregate peak memory разных thread configurations.

## Этап 6. CLI, документация и переносимость

### 6.1 CLI

Файлы: `src/mimosa/cli.py`, `tests/test_cli.py`.

- [ ] Добавить `build-null --background` и передавать batch в `build_null`.
- [ ] Требовать положительные ограниченные `--num-sequences`, `--seq-length` и
  `--num-samples` до allocation.
- [ ] Отклонять неиспользуемые options: FASTA/background для scores-only,
  null-distribution/effective targets без `--pvalue`.
- [ ] Убрать дублирование имени exception в stderr.
- [ ] Явно решить, читает ли `build-null` первый motif или все motifs из MEME;
  реализовать выбранный контракт и тест.

### 6.2 Документация и примеры

Файлы: `README.md`, `docs/python/*.md`, `examples/run.ps1`.

- [ ] Исправить `run.ps1`: `profile` -> `compare`, старые model options ->
  `--query-type`/`--target-type`.
- [ ] Передавать `sequences` в parallel quickstart с raw targets.
- [ ] Удалить заявление о legacy pickle fallback и ложноположительный тест.
- [ ] Синхронизировать `SiteCollection.to_dict`, error classes/exit codes и
  обязательное поле `n_sites`.
- [ ] Описать физическую семантику offsets после исправления context alignment.

### 6.3 Packaging, CI и release

Файлы: `pyproject.toml`, `.github/workflows/test.yml`,
`.github/workflows/publish.yml`, `src/mimosa/io/bundles.py`,
`src/mimosa/__init__.py`.

- [ ] Не анализировать host `sys.argv` и не менять `NUMBA_NUM_THREADS` при
  обычном `import mimosa`; перенести CLI bootstrap в entrypoint.
- [ ] Исправить ложную provenance `Mimosa.jl 0.1.0` на фактический Python tool
  version либо убрать её.
- [ ] Добавить license metadata и project URLs.
- [ ] Либо добавить Windows CI и актуальный PowerShell smoke test, либо явно
  снять обещание Windows support и удалить Windows-only surface.
- [ ] Ограничить поддерживаемый диапазон Python или добавить текущую новую
  версию Python в CI.
- [ ] В release workflow проверить wheel и sdist, включая console entrypoint;
  минимизировать permissions и по возможности закрепить actions по SHA.

## Этап 7. Удалить устаревший код и документы

- [ ] Удалить завершённый `PLAN.md` после подтверждения, что миграция полностью
  отражена в текущем коде и документации.
- [ ] Удалить внутренний `cache_get`, если внешняя совместимость не заявлена.
- [ ] Удалить ложный legacy pickle test и документацию, не добавляя pickle load.
- [ ] Удалить hard-coded normalization/version literals, используя owning
  constants и `normalization_fingerprint`.
- [ ] Не удалять `joblib`, `numba` или `numpy`: у всех зависимостей есть
  подтверждённое production-использование.

Ожидаемое упрощение: около 101 строки после удаления старого плана и ещё около
29 строк условно при удалении `cache_get` и legacy pickle surface; зависимости
не удаляются.

## Финальная проверка

- [ ] `uv run ruff check .`
- [ ] `uv run pytest -q`
- [ ] `uv run bash examples/run.sh`
- [ ] PowerShell/Windows smoke test, если Windows остаётся поддерживаемой.
- [ ] `uv run python benchmarks/benchmark_performance.py --output /tmp/mimosa-performance-after.json`
- [ ] Сравнить before/after medians только для одинаковых workloads и thread
  budgets.
- [ ] Проверить JSON-only `stdout`, диагностику в `stderr`, target order и
  serial/joblib численную эквивалентность для всех metrics и threshold modes.
- [ ] Очистить тестовые cache/build artifacts и убедиться, что они не попали в
  version control.

## Порядок поставки

1. P0 отдельными небольшими изменениями: cache clear, empirical lookup,
   coordinates, immutable invariants, bundle confinement.
2. P1 correctness/API одним или несколькими независимыми изменениями.
3. Быстрые оптимизации с отдельными performance regression checks.
4. Исправление benchmark до более глубоких оптимизаций кэша и alignment.
5. CLI/docs/CI и удаление устаревшего surface.

Работа считается завершённой только после прохождения финальной проверки и
получения измерений, подтверждающих, что оптимизации не ухудшили корректность,
пиковую память или производительность ключевых workloads.
