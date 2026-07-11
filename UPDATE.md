# План оптимизации MIMOSA для сценария one-to-many

## 1. Цель и ограничения

Целевой сценарий производительности:

- одна query-модель сравнивается примерно с 500 target-моделями;
- используются 10 000 последовательностей длиной 100 нуклеотидов;
- основной интерес представляет `profile` workflow, включая сканирование обеих цепей, нормализацию профилей,
  выбор anchors и alignment;
- один процесс Python живет достаточно долго, поэтому стоимость JIT-компиляции допустима один раз на сигнатуру;
- Numba disk cache использовать нельзя: все kernels должны оставаться с `cache=False`, поскольку рабочая среда может
  не предоставлять доступ на запись к исходному дереву, cache directory или домашнему каталогу;
- результаты, tie-breaking, offsets, orientations и порядок targets должны оставаться детерминированными;
- оптимизации нельзя оплачивать неограниченным ростом памяти: одновременное хранение профилей всех 500 targets не
  допускается.

Итоговая цель: ускорить warm one-to-many workflow, сохранив текущий публичный API и численное поведение в пределах
уже принятых допусков. Cold JIT time измеряется отдельно и не смешивается с warm throughput.

## 2. Наблюдаемая проблема

Текущая реализация использует разные уровни параллелизма несогласованно:

- `_run_target_comparisons()` в `src/mimosa/comparison/runner.py` обрабатывает targets последовательно и игнорирует
  `n_jobs` через `del n_jobs`;
- kernels сканирования в `src/mimosa/functions/scanning.py` используют обычный `range`, не объявлены с
  `parallel=True` и поэтому не реагируют на `set_num_threads()`;
- `prange` применяется только в profile alignment;
- alignment запускает отдельный parallel region для каждого shift и orientation, создавая до 44 запусков на target
  при `search_range=5` и четырех orientations;
- query bundle переиспользуется, но query anchors сейчас могут вычисляться заново для каждого target;
- новый `target_cache` создается для каждого target, поэтому fingerprint одних и тех же sequence/background batches
  может вычисляться до 500 раз;
- одинаковые по длине последовательности все равно сопровождаются dense masks и промежуточными копиями strand
  arrays.

Изменения следует вносить небольшими этапами. После каждого этапа необходимо измерять полный workflow, а не только
изолированный kernel.

## 3. Этап 0: воспроизводимый baseline

### 3.1. Добавить benchmark полного workflow

Создать `benchmarks/profile_one_to_many.py`, который:

- генерирует или загружает один фиксированный query и набор targets;
- использует batch из 10 000 последовательностей длиной 100;
- поддерживает как минимум 1, 10, 100 и 500 targets;
- принимает `--threads`, `--repeats`, `--seed`, `--sequence-count`, `--sequence-length` и `--target-count`;
- отдельно выводит cold time, первый warm run и медиану последующих warm runs;
- измеряет время стадий: query preparation, target scan, normalization, anchor preparation, alignment и total;
- сообщает peak RSS, размеры основных массивов и количество Numba threads;
- вычисляет checksum результатов, чтобы benchmark не мог пропустить работу из-за ошибки или будущей оптимизации;
- не включает генерацию входных данных и импорт модулей в warm measurement.

Расширить `benchmarks/threadscaling.py` отдельными режимами для scanning, log-tail mapping и alignment. Проверять
`1, 2, 4` потока и, только как дополнительный результат, SMT thread count.

### 3.2. Зафиксировать baseline

Для каждого benchmark сохранить в отчете:

- Python, NumPy, Numba и Joblib versions;
- CPU model, число физических и логических ядер;
- Numba threading layer;
- параметры модели: тип, `kmer`, motif length;
- параметры comparison: metric, search range, window radius, threshold и orientations;
- cold/warm time, peak RSS и checksum.

Gate этапа: benchmark воспроизводится тремя последовательными запусками, warm median отличается не более чем на 10%,
а checksum результатов стабилен.

## 4. Этап 1: убрать повторную работу в one-to-many

Этот этап не меняет parallel kernels и должен быть реализован первым, поскольку снижает работу независимо от числа
ядер.

### 4.1. Переиспользовать fingerprints входных batches

Затрагиваемые файлы:

- `src/mimosa/comparison/common.py`;
- `src/mimosa/comparison/profile.py`;
- `src/mimosa/comparison/motif.py`.

Изменения:

1. На уровне `_compare_profile_one_to_many()` один раз вычислить fingerprints для `sequences` и `background`.
2. Не помещать batch fingerprints только в локальный `target_cache`, который уничтожается после каждого target.
3. Ввести небольшой immutable preparation context или общий read-only cache с batch fingerprints. Не использовать
   глобальное состояние.
4. Target-specific cached arrays продолжать освобождать после target, чтобы память оставалась ограниченной.
5. Аналогично проверить motif one-to-many/PFM path.

Тесты:

- monkeypatch/spy подтверждает, что `fingerprint_batch(sequences)` вызывается O(1), а не O(number of targets);
- результаты и порядок targets совпадают с текущей реализацией;
- общий cache не удерживает target bundles после завершения target.

### 4.2. Подготовить query anchors один раз

В `src/mimosa/comparison/profile.py` разделить подготовку query и target:

1. После построения `query_bundle` один раз вычислить anchors для необходимых query strands.
2. Один раз преобразовать их в CSR через `build_anchor_csr()`.
3. Передавать готовые query anchors в target scorer.
4. Для каждого target вычислять только target anchors.
5. Не менять существующий deterministic ordering внутри CSR.

Тесты:

- spy подтверждает один вызов query anchor preparation на one-to-many operation;
- best-anchor и threshold-anchor режимы дают прежние результаты;
- пустые и короткие sequences остаются безопасными;
- порядок anchors и tie-breaking не меняются.

Gate этапа: полная тестовая suite проходит; warm time не ухудшается; число повторных fingerprint и query-anchor
операций не зависит от количества targets.

## 5. Этап 2: распараллелить sequence scanning

Затрагиваемые файлы:

- `src/mimosa/functions/scanning.py`;
- `src/mimosa/scanning.py`;
- `tests/unit_functions.py` и/или `tests/unit_comparison.py`;
- benchmark-файлы.

### 5.1. Сохранить отдельные serial и parallel kernels

Не заменять единственный kernel безусловным `parallel=True`. Создать serial/parallel варианты либо общий inline helper
и два внешних kernels:

- forward scan;
- reverse scan;
- fused both-strand scan.

Parallel kernels объявлять как `@njit(cache=False, parallel=True, fastmath=True)`. Цикл
`for row_index in prange(n_rows)` безопасен, поскольку каждая итерация читает отдельную sequence row и пишет только
в соответствующую output row. Внутренние циклы по positions, terms и k-mer offsets оставить serial.

Fused both-strand kernel является основным путем для profile comparison. Он должен вычислять forward и reverse scores
за один проход по строке, чтобы повторно использовать sequence row и не создавать два независимых batch calls.

### 5.2. Добавить измеряемый dispatch threshold

Добавить функцию наподобие `should_parallelize_scan(...)`, учитывающую:

```text
work = n_rows * max_scores * n_terms * kmer * n_strands
```

Parallel path выбирать только при `num_threads > 1` и превышении порога. Первичное значение порога определить
benchmark-ом, а не копировать порог alignment. Добавить тесты непосредственно ниже и выше порога.

### 5.3. Проверить layout и allocations

В рамках этого этапа:

- формировать итоговый strand layout `(2, rows, width)` непосредственно или возвращать views без `np.stack`;
- не копировать `values`, если dtype и C-contiguity уже подходят;
- не менять публичную структуру `ProfileBundle`;
- отдельно измерить стоимость mask, но не удалять ее до этапа 5, чтобы не смешивать оптимизации.

Тесты и свойства:

- parallel output точно совпадает с serial output для scores, masks и lengths;
- forward/reverse coordinates совпадают на PWM, BaMM, SiteGA, Dimont и Slim fixtures;
- одинаковый результат при 1, 2 и 4 threads;
- ragged, empty, too-short и ambiguous-base inputs;
- input arrays не изменяются;
- fused both-strand output совпадает с двумя отдельными strand scans.

Gate этапа: на 10 000×100 fused scanner ускоряется на 4 физических ядрах, а полный one-to-many workflow не
регрессирует. Порог принимается только на основе warm measurements.

## 6. Этап 3: распараллелить empirical log-tail mapping

Затрагиваемые файлы:

- `src/mimosa/functions/tails.py`;
- tests для normalization и profile comparison;
- benchmarks.

### 6.1. Parallel mapping по независимым строкам

Добавить serial и parallel варианты `_apply_score_log_tail_table_numba`. Безопасная единица работы -- одна строка:

```python
for row_index in prange(rows):
    for col_index in range(cols):
        ...
```

Каждая итерация пишет в отдельную строку `mapped`; lookup table доступна только для чтения. `_lower_bound_desc`
оставить inline/serial helper.

### 6.2. Обработать обе цепи одним kernel call

Добавить bundle kernel для `(strands, rows, width)` и распараллеливать плоский независимый индекс
`strand_index * rows + row_index`. Это устраняет Python-loop по strands и уменьшает количество Numba dispatches.

### 6.3. Dispatch threshold

Порог должен учитывать число валидных значений и примерную стоимость бинарного поиска:

```text
work ~= valid_values * log2(table_length)
```

Не включать parallel path для малых таблиц и профилей без benchmark-подтверждения.

Тесты:

- serial/parallel exact equality для mapped `float32` values;
- одинаковая обработка граничных значений таблицы, duplicates, padding и empty table;
- одинаковый результат для dense и ragged inputs;
- одинаковый результат при разных thread counts.

Gate этапа: mapping ускоряется на целевой форме данных, full-workflow checksum не меняется.

## 7. Этап 4: оптимизировать anchors

Затрагиваемый файл: `src/mimosa/comparison/profile.py`, при необходимости выделить kernels в отдельный модуль.

### 7.1. Best-anchor mode

Best anchor для каждой sequence независим. Parallel kernel может писать результат непосредственно по `row_index`.
Для строк нулевой длины использовать validity array или sentinel, затем выполнить стабильную serial compaction.
Для обычного случая одинаковых непустых последовательностей compaction не нужна.

### 7.2. Threshold-anchor mode

Нельзя использовать общий mutable `out_index` внутри `prange`. Реализовать безопасный двухпроходный алгоритм:

1. parallel count anchors для каждой строки;
2. deterministic serial prefix sum offsets;
3. parallel fill, где каждая строка пишет только в свой диапазон;
4. сохранить row-major ordering текущей реализации.

Порог parallel dispatch определить отдельно для best и threshold modes. При редких anchors двухпроходный parallel
алгоритм может быть медленнее serial.

Тесты:

- exact equality rows/positions между serial и parallel;
- stable row-major order;
- ties выбирают первую позицию, как сейчас;
- threshold equality включается по прежнему правилу;
- empty rows и отсутствие anchors.

Gate этапа: anchors ускоряются на 10 000 rows без изменения порядка или результатов alignment.

## 8. Этап 5: сократить parallel launches и allocations alignment

Затрагиваемые файлы:

- `src/mimosa/functions/alignment.py`;
- `src/mimosa/comparison/profile.py`;
- alignment tests и benchmarks.

### 8.1. Переиспользовать workspace

Сейчас workspace создается внутри orientation scoring. Изменить lifetime:

- выделять workspace один раз на target comparison и переиспользовать между orientations;
- если shapes query/target различаются, использовать максимальную необходимую ширину или безопасно пересоздавать;
- не переносить workspace между одновременно исполняемыми targets;
- проверить integer generation overflow и предусмотреть сброс marks перед переполнением.

### 8.2. Объединить shifts в один parallel launch

Текущая схема запускает `prange` отдельно для каждого shift. Спроектировать kernel, который обрабатывает все shifts
одной orientation за один вызов. Предпочтительный layout partial reductions:

```text
partials[shift_index, row, statistic]
```

Возможные варианты распараллеливания необходимо сравнить benchmark-ом:

- `prange` по rows, внутренний serial loop по shifts;
- `prange` по flattened `(shift, row)`.

Первый вариант лучше переиспользует row data и уменьшает scheduling overhead; второй дает больше независимых задач,
но увеличивает working set. Reduction по rows выполнять детерминированно в фиксированном порядке после kernel.

### 8.3. Не распараллеливать orientations вложенным `prange`

Numba nested parallel regions здесь не нужны. Orientations мало, а основной параллелизм уже находится по строкам.
Избегать одновременного target-level и inner-kernel parallelism до отдельного этапа 6.

Тесты:

- точное совпадение `n_sites`, shift и orientation;
- floating scores сравниваются существующими обоснованными tolerances;
- deterministic reductions при 1, 2 и 4 threads;
- все metrics и anchor modes;
- search ranges 0, 1 и типичное значение 5;
- повторное использование workspace не оставляет данные предыдущего target/orientation.

Gate этапа: количество Numba parallel launches на target существенно сокращено; peak RSS остается приемлемым;
one-to-many warm throughput улучшается.

## 9. Этап 6: выбрать один уровень параллелизма для 500 targets

Этот этап выполняется только после оптимизации внутренних kernels.

Сравнить две политики:

### Политика A: один процесс, parallel Numba kernels

- targets обрабатываются последовательно;
- scanning, normalization, anchors и alignment используют общий Numba thread budget;
- query state естественно переиспользуется без сериализации;
- peak memory ограничен профилем одного target.

Это базовая и предпочтительная политика, если она эффективно загружает физические ядра.

### Политика B: Joblib processes по target chunks

- вернуть process-level parallelism только для крупных one-to-many collections;
- внутри workers установить Numba threads в 1;
- передавать targets chunks, а не по одной задаче, чтобы амортизировать IPC;
- использовать read-only memmap/shared representation для query bundle и sequence batch, если Joblib не делает это
  автоматически и копирование заметно по RSS;
- сохранять входной порядок результатов;
- не хранить подготовленные профили всех targets одновременно;
- worker initializer должен один раз прогреть необходимые Numba signatures, поскольку `cache=False`;
- число процессов по умолчанию ограничить физическими ядрами и доступной памятью.

Запрещенная политика: несколько Joblib processes, каждый из которых использует все Numba threads. Это приводит к
oversubscription и нестабильной производительности.

Выбор политики должен зависеть от workload:

- single pair или небольшое число targets: политика A;
- сотни targets: выбрать A или B по полному benchmark, RSS и стоимости worker warm-up;
- не менять значение `n_jobs` молча: документировать, означает ли оно Numba threads или target workers, либо разделить
  настройки на `threads` и `workers` с совместимым переходным поведением.

Gate этапа: выбранная политика быстрее альтернативы на 500 targets, не превышает установленный memory budget и дает
идентичные упорядоченные результаты.

## 10. Этап 7: оптимизации памяти и layout

Выполнять только после profiling, чтобы не смешивать layout changes с parallel correctness.

Кандидаты:

1. Для batch с одинаковыми lengths использовать implicit prefix validity вместо dense mask внутри hot path.
2. Сохранять публичный mask при необходимости только на API boundary.
3. Использовать `int32` для lengths/offsets только после явной проверки максимального размера и без неоднозначных
   conversions.
4. Возвращать оба strands сразу в `ProfileBundle` layout, исключив промежуточные `MaskedBatch` и `np.stack`.
5. Избегать `np.ascontiguousarray` там, где dtype/layout уже гарантированы вызывающим кодом.
6. Не выделять masks и arrays повторно для каждого orientation.

Для каждого изменения измерять allocations и peak RSS. Экономия памяти не должна менять публичный формат или
поведение malformed-input validation.

## 11. Что не следует распараллеливать

- Короткий matrix/tensor motif alignment по offsets: motifs обычно малы, и `prange` overhead превысит вычисления.
- Tie-breaking и выбор best result: эти операции короткие и должны оставаться serial/deterministic.
- Prefix sums для threshold anchors: serial prefix sum дешев и задает детерминированные output ranges.
- Fingerprint одного массива: сначала устранить повторное хеширование; parallel hashing усложнит код без достаточной
  пользы.
- Одновременно targets, orientations, shifts и rows: выбирать один крупный уровень плюс один внутренний уровень без
  nested oversubscription.

## 12. Проверка корректности

После каждого этапа выполнять:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Минимальный набор performance/correctness проверок:

- serial и parallel kernels на одинаковых inputs;
- `NUMBA_NUM_THREADS`/`set_num_threads` для 1, 2 и 4 threads;
- PWM, BaMM, SiteGA, Dimont и Slim scanning paths;
- forward, reverse, best и both strand modes;
- equal-length и ragged batches;
- empty, short и ambiguous sequences;
- best-anchor и threshold-anchor modes;
- все profile metrics;
- один target и 500 targets;
- одинаковый target ordering и deterministic tie-breaking;
- отсутствие мутации входных models, sequences и background;
- bounded memory: target-specific state освобождается после каждого target/chunk.

Если `fastmath=True` уже используется, parallel/serial equivalence проверять на frozen fixtures и не ослаблять
tolerances только ради прохождения тестов. Любое изменение accumulation order документировать отдельно.

## 13. Критерии завершения

Работа считается завершенной, когда:

- `cache=False` сохранен для всех Numba kernels и запуск не требует writable cache directory;
- query scan, query normalization, query anchors и batch fingerprints выполняются O(1) раз на one-to-many operation;
- scanner и log-tail mapping используют безопасный parallel path на целевой форме данных;
- alignment не создает отдельный parallel region без необходимости для каждого мелкого фрагмента работы;
- thread/process policy исключает oversubscription;
- результаты совпадают с текущим контрактом и детерминированы между thread counts;
- full one-to-many benchmark для 500 targets показывает улучшение warm wall time;
- benchmark report содержит cold/warm timings, peak RSS, environment и checksum;
- unit, integration, lint и format checks проходят.

## 14. Рекомендуемый порядок pull requests

1. Benchmark harness и baseline report без production changes.
2. One-to-many fingerprints и query-anchor reuse.
3. Parallel fused scanner с serial fallback.
4. Parallel bundle log-tail mapping.
5. Parallel anchor preparation.
6. Alignment workspace reuse и fused shifts.
7. Target-level policy benchmark и, при необходимости, chunked Joblib workers.
8. Отдельный PR для mask/layout оптимизаций.

Каждый PR должен содержать отдельные correctness results и performance results. Не объединять изменение численного
порядка reductions, parallel strategy и layout representation в один непрозрачный commit.
