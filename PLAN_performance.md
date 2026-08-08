# План оптимизации производительности

## Ограничения

- Основной API остаётся one-query: `compare_many(query, targets, ...)`.
- За один запуск используется одна строковая `metric`; список метрик в API и CLI не добавляется.
- Порядок результатов `compare_many` остаётся стабильным.
- Process pool не возвращается. Параллелизм сравнения остаётся внутри Numba.
- Дисковый cache сохраняет атомарную запись и проверку целостности.
- Сначала измеряется текущая версия, затем принимается только оптимизация с подтверждённым выигрышем.

## Базовая точка

Добавить воспроизводимый benchmark для production-пути:

- `10_000` foreground-последовательностей;
- реальный background FASTA;
- `64`, `128` и `256` targets;
- `min_logerr=2`;
- cold cache, disk hit и memory hit;
- `NUMBA_NUM_THREADS=1,2,4,6,8`.

Для каждого прогона сохранять:

- время подготовки профилей;
- время cache read, checksum и decode;
- время normalization и anchors;
- время packing и alignment kernel;
- peak RSS и размер cache на диске;
- число Numba threads и выбранный dispatch path.

Критерий сравнения: median минимум трёх повторов после отдельного JIT warmup.

## 1. Ускорить холодную подготовку профилей

### Проблема

Подготовка профиля включает scan, normalization, anchor collection, pickle и
запись в cache. На production-sized данных cold path доминирует над самим
сравнением.

### Правки

- Вынести фазовые таймеры в benchmark, не добавляя постоянный logging в hot path.
- Сравнить стоимость `_scan_models_batch`, `_fit_normalize`, `collect_both_anchors`, `_encode_prepared_profile` и `cache_set`.
- Оптимизировать только фазу, которая занимает большую часть cold path.
- Сохранить текущую batch-подготовку targets через `_prepare_profiles_batch`.

### Файлы

- `src/mimosa/profiles/prepared.py`
- `src/mimosa/scan.py`
- `src/mimosa/profiles/normalization.py`
- `src/mimosa/profiles/anchors.py`
- `src/mimosa/cache.py`

### Проверка

- Результаты до и после совпадают с текущими в тестах и на production fixture.
- Cold wall time и peak RSS сравниваются отдельно; улучшение одного не должно скрывать ухудшение другого.

## 2. Адаптивный dispatch для малых групп scan

### Проблема

`_scan_models_batch` использует kernels с `prange` по моделям. Для группы из
одной-двух моделей такая распараллелка недогружает CPU, а накладные расходы
многопоточности могут быть выше выигрыша.

### Правки

- Оставить текущий model-parallel kernel для больших групп одного типа и одинаковой геометрии.
- Для малых групп использовать serial kernel либо row-parallel `batch_scan_*_parallel` по числу строк.
- Вынести порог в существующую политику `use_parallel`, не вводить отдельную систему настроек.
- Для custom models сохранить текущий serial путь.
- Не запускать вложенный `prange`: одновременно распараллеливать models и rows нельзя без риска oversubscription.

### Файлы

- `src/mimosa/scan.py`
- `src/mimosa/parallel.py`
- `src/mimosa/_kernels.py`

### Проверка

- Численная эквивалентность serial и parallel scan.
- Benchmark групп размером `1`, `2`, `4`, `16`, `64` на реальном размере sequence batch.
- Проверить, что small workloads не становятся медленнее serial path.

## 3. Mmap-friendly формат дискового cache

### Проблема

Текущий disk hit делает `read -> SHA-256 -> pickle.loads`. Это повторно копирует
и декодирует крупный профиль, даже если он уже был создан корректно.

### Правки

- Добавить новую версию бинарного формата prepared profile с фиксированными секциями для score data, offsets и anchors.
- Хранить в metadata формат, dtype, размеры и offsets секций.
- Читать числовые массивы через read-only `mmap`/`numpy.memmap`, чтобы не делать pickle decode и лишнюю копию.
- Сохранить atomic stage + rename при записи.
- Увеличить `ALGORITHM_VERSIONS["prepared_profile"]`, чтобы старые записи не трактовались как новый формат.
- Оставить fallback на старый pickle-формат для cache entries, созданных предыдущей версией, либо явно считать их cache miss.
- Проверять checksum при первом открытии entry; не отключать проверку целостности молча.
- Проверить lifetime mmap-backed arrays и совместимость с `PreparedProfile`/pickle.

### Файлы

- `src/mimosa/cache.py`
- `src/mimosa/profiles/prepared.py`
- `src/mimosa/arrays.py`
- `tests/test_cache.py`
- `tests/test_profiles.py`

### Проверка

- Disk-hit должен возвращать те же scores, offsets, anchors и результаты сравнения.
- Проверить повреждённый файл, обрыв записи, старую версию cache и удаление cache.
- Сравнить disk-hit time и RSS с текущим pickle-путём.

## 4. Byte-budgeted LRU вместо лимита по количеству

### Проблема

`_MEMORY_CACHE_MAX_PROFILES=256` не учитывает, что профили имеют разный размер.
Размер target-блока и лимит LRU также являются разными настройками.

### Правки

- Заменить лимит количества на лимит bytes для `Cache._prepared_profiles`.
- Считать размер `PreparedProfile` по backing arrays: scores, offsets и anchors.
- Перед вставкой вытеснять старые entries до соблюдения бюджета.
- Не увеличивать бюджет до всех `1595` targets: это требует примерно `11-12 GB` только под profile payload.
- Сделать бюджет настраиваемым для benchmark/production, не меняя CLI metric API.
- Начальный benchmark диапазон: `512 MiB`, `1 GiB`, `2 GiB`; выбрать default по peak RSS и swap pressure.
- Target batch size проверять отдельно: его нельзя автоматически приравнивать к LRU budget.

### Файлы

- `src/mimosa/cache.py`
- `tests/test_cache.py`

### Проверка

- Тесты eviction по bytes, обновления LRU при hit и entries, превышающей весь budget.
- Проверить отсутствие неконтролируемого роста RSS.
- Сравнить hit rate для последовательного прохода по `1595` targets.

## 5. Снизить стоимость target packing

### Проблема

`_compare_many_prepared_parallel` собирает target arrays через
`np.concatenate`. Это необходимо для текущего kernel ABI, но создаёт копии и
временный peak memory.

### Правки

- Сначала измерить долю packing в полном времени `compare_many` и его peak RSS.
- Если доля существенна, заменить список частей + несколько `concatenate` на заранее рассчитанные offsets и один preallocated buffer с `copyto`.
- Сохранить текущий special case для shared forward/reverse strands.
- Не создавать reusable global packed cache для one-query API без подтверждённой пользы.
- Не менять Numba kernel ABI до отдельного benchmark: packing может быть дешевле изменения kernel.

### Файлы

- `src/mimosa/compare.py`
- `src/mimosa/_kernels.py` только при необходимости
- `tests/test_profiles.py`

### Проверка

- Сравнить результаты serial/parallel и все три поддерживаемые metric.
- Измерить packing time, peak RSS и время kernel отдельно.
- Проверить target batches меньше и больше `MIN_PARALLEL_TARGETS`.

## 6. Уточнить блокировки cache

### Текущее состояние

Process pool в runtime удалён. В обычном one-process запуске текущий cache-wide
`fcntl.flock` не является измеренным bottleneck. `multiprocessing` используется
только в тесте cache lock.

### Правки

- Не возвращать process pool и не добавлять thread pool.
- Не менять lock до появления подтверждённого concurrent-writer сценария.
- Если появится несколько writer processes, заменить cache-wide write lock на per-key lock с сохранением atomic rename.
- Оставить общий lock только для `clearcache`/операций над списком entries.
- Проверить гонки: два writer одного key, два writer разных keys, reader во время rename.

### Файлы

- `src/mimosa/cache.py`
- `tests/test_cache.py`

## Порядок внедрения

1. Добавить baseline benchmark и фазовые измерения.
2. Проверить адаптивный scan dispatch для малых групп.
3. Ввести byte-budgeted LRU и выбрать бюджет по RSS.
4. Оптимизировать packing, только если benchmark подтвердит его стоимость.
5. Реализовать mmap-friendly cache format как отдельную версию формата.
6. Per-key locks оставить отложенными до подтверждения multi-process записи.

После каждого шага запускать:

```bash
PYTHONPATH=src NUMBA_NUM_THREADS=6 python -m pytest -q
```

Финальный критерий: ускорение production cold/warm path без изменения
результатов, one-query `compare_many`, CLI и формата одной metric на запуск.
