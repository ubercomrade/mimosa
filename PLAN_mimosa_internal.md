# План переноса оптимизаций внутрь mimosa

## Цель

Перенести всю orchestration-оптимизацию из серверного скрипта в пакет
`mimosa`, сохранив единую публичную точку сравнения `compare_many()`.

За один вызов используется ровно одна метрика. Поддержка iterable метрик,
fused multi-metric kernels и возврат результатов для нескольких метрик в одном
вызове в этот план не входят.

Основной сценарий:

```text
1579 query x 1595 target x one metric x 4 min_logerr
10 000 sequences x 100 bp
```

## Ограничения и контракт

- Сигнатуры `compare_many()` и `prepare_profile()` не менять без отдельного
  согласования.
- `metric` остается строковым кодом: `co`, `dice` или `cosine`.
- `compare_many()` остается единой публичной точкой сравнения.
- Не добавлять публичные API для target shards, worker sessions или cache
  contexts.
- Сохранять порядок targets, duplicate targets и существующий tie-breaking.
- Не менять математическую семантику метрик.
- Не использовать nested parallelism.
- Pickle-cache directory считается доверенным, но checksum при disk cache hit
  сохраняется.
- Custom `MotifModel.scan_into()` остается последовательным fallback.

## Текущее состояние

- `EncodedSequences` уже хранит batch sequences в flat buffer с offsets.
- Scan и normalization имеют Numba kernels с выбором serial/parallel path.
- Подготовленные targets размером от 64 элементов сравниваются packed Numba
  kernel с parallel по targets.
- Неподготовленные большие списки targets все еще используют ProcessPool.
- Target batches формируются серверным скриптом, а не `mimosa`.
- Disk cache не имеет in-memory слоя: каждый hit читает pickle, проверяет
  checksum и декодирует профиль.
- Fingerprint sequences не переиспользуется между отдельными вызовами
  подготовки profiles.

## 1. Внутренние bounded target batches

### Задача

`compare_many()` должен принимать полный список targets и сам ограничивать
размер рабочего batch. Серверный скрипт не должен управлять target shards.

### Реализация

Добавить приватную orchestration-логику:

```text
targets
  -> bounded slices
  -> prepare current slice
  -> packed comparison
  -> append ordered results
  -> release current slice
```

- Использовать внутренний default batch size, например 256.
- Не добавлять публичный API для shards.
- При необходимости настройки вынести во внутреннюю policy/env setting, не
  меняя обязательный публичный контракт.
- Поддержать targets, которые уже являются `PreparedProfile`, без повторной
  подготовки.
- Для неподготовленных targets готовить только текущий slice.
- Прогресс считать относительно полного списка targets.
- Результаты собирать в исходном порядке, даже если внутренний kernel или
  fallback завершает элементы в другом порядке.
- Для пустого списка вернуть пустой список.

### Server integration

Упростить `compare_hocomoco_server.py`:

- убрать внешний цикл по target batches;
- передавать полный target list в `compare_many()`;
- оставить в скрипте только загрузку данных, resume/output и запуск одной
  метрики;
- не держать одновременно весь набор prepared targets в памяти скрипта.

## 2. Ограниченный in-memory cache

### Задача

Убрать повторное чтение и декодирование одного prepared profile внутри процесса.

### Реализация

В `Cache` добавить приватный bounded cache:

```text
content-addressed key -> PreparedProfile
```

- Проверять in-memory cache до `cache_get()`.
- После успешной подготовки и disk store сохранять тот же объект в memory.
- После disk cache hit сохранять валидированный объект в memory.
- Повторный hit должен возвращать объект без read, SHA-256 и pickle decode.
- Использовать LRU или другой bounded policy с ограничением по числу профилей
  или bytes.
- Не кешировать fingerprint по `id()` изменяемого массива.
- Разные процессы имеют независимые memory caches.
- Добавить приватный способ очистки/освобождения entries после target batch.
- Не удалять disk cache и не ослаблять checksum validation.

### Validation

Обычный `PreparedProfile` constructor остается validating path для:

- ручного создания;
- pickle payload;
- внешних источников.

Trusted constructor используется только после внутреннего pipeline
`RaggedArray -> normalize -> collect anchors`.

## 3. Внутренний preparation context

### Задача

Вычислять fingerprint sequences и background один раз на orchestration-вызов,
а не на каждый target.

### Реализация

Добавить приватный context, не экспортируя его в public API:

```text
PreparationContext:
  sequences
  sequence_fingerprint
  background
  background_fingerprint
```

- Создавать context внутри `compare_many()`.
- Если `background is sequences`, использовать один fingerprint для обоих
  полей.
- Для отдельного background вычислять его fingerprint один раз.
- Передавать context во внутренние cache-key и preparation helpers.
- Публичный прямой вызов `prepare_profile()` должен работать без context.
- Cache key bytes и состав key не менять.

### Проверки

- fingerprint call count для 1 query + N targets;
- идентичность key до и после оптимизации;
- отдельные sequences/background;
- одинаковый объект sequences как foreground и background;
- отсутствие cache по `id()` mutable arrays.

## 4. Подготовка targets без ProcessPool

### Задача

Убрать ProcessPool из основного пути, когда targets можно подготовить и
сравнить в bounded batch.

### Реализация

Для каждого target batch:

1. Подготовить profiles в родительском процессе.
2. Переиспользовать prepared query.
3. Упаковать текущие targets.
4. Запустить один Numba comparison kernel.
5. Освободить packed arrays и временные profiles, если они не удерживаются
   bounded memory cache.

ProcessPool оставить только как явно обоснованный fallback для окружений или
моделей, где невозможно применить универсальный path. Fallback не должен
запускать Numba parallel kernels внутри worker processes.

## 5. Universal packed comparison

### Packed representation

Для каждого target batch подготовить:

- flat forward score data;
- flat reverse score data;
- offsets scores по target и row;
- flat forward/reverse anchor positions;
- offsets anchors по target и row;
- `target index -> original target name`;
- flags shared forward/reverse bundle.

Не создавать отдельные копии shared strand без необходимости.

### Kernels

Для `min_logerr <= 0` использовать best-anchor kernel:

```text
prange(target_index)
  orientation
    shift
      score alignment
```

Для положительного threshold использовать независимое scratch state:

- per-target buffers;
- либо per-thread buffers;
- либо двухпроходный алгоритм без общего `seen/candidates`.

Запрещено использовать один scratch buffer для всех `prange` targets.

### Policy

- Большой batch: parallel по targets.
- Малый batch: serial path или parallel по rows, если это измеримо выгодно.
- Не совмещать parallel по targets и rows.
- Учитывать `NUMBA_NUM_THREADS`.
- При недоступности parallel threading сохранять корректный serial fallback.

### Correctness

Сравнивать packed path с текущим serial path для:

- `co`, `dice`, `cosine` по отдельным вызовам;
- `min_logerr <= 0`;
- положительных thresholds;
- всех четырех orientations;
- shared strands и mixed strand bundles;
- duplicate targets;
- target order;
- ties по score, number of sites, shift и orientation rank;
- разных `NUMBA_NUM_THREADS`.

Проверять:

- scores через `assert_allclose`;
- exact offset/orientation/n_sites;
- одинаковый `ComparisonResult.to_dict()`.

## 6. Batch scan для встроенных моделей

### Общий private contract

Добавить внутренний dispatcher:

```text
models + EncodedSequences
  -> packed raw forward/reverse profiles
  -> per-model geometry metadata
```

Требования:

- сохранять исходный порядок models;
- поддерживать ragged offsets;
- не смешивать score data разных models;
- не менять public `MotifModel.scan_into()` contract;
- выбирать kernel и geometry до запуска Numba;
- не запускать существующий parallel kernel отдельно внутри outer `prange`.

### PWM

- Группировать модели по motif length либо использовать padded weights + lengths.
- Добавить forward и reverse batch kernels.
- Поддержать models длины 1 и максимальной длины.
- Проверить N bases и strand identity.

### BaMM, Dimont, Slim

- Группировать по `(order, motif_length)`.
- Передавать packed weights одной geometry-группы в rolling kernel.
- Сохранять отдельные forward/reverse paths.
- Не объединять несовместимые geometry.
- Проверить order 0, максимальный order и разные motif lengths.

### SiteGA

- Использовать отдельный dispatch.
- Сохранить `kmer=2` и текущую coordinate contract.
- Проверить минимальную длину, offsets и forward/reverse scores.

### Custom models

- Оставить текущий последовательный `scan_into()` fallback.
- Не добавлять обязательный `batch_scan_into()` capability.
- Расширение для custom models добавлять только при реальном потребителе.

## 7. Batch normalization и profile creation

Внедрять после batch scan:

1. Сначала использовать существующие normalization paths без изменения
   математической семантики.
2. Профили создавать через trusted fast path после полной внутренней
   валидации pipeline.
3. При подтвержденном bottleneck добавить parallel normalization по model
   index.
4. Не объединять calibration distributions разных models.
5. Для каждой model сохранять собственные normalization table и anchors.

## 8. Memory и lifecycle

- Ограничивать target batches внутри `compare_many()`.
- Не удерживать весь набор targets в RAM.
- Не создавать копии packed arrays без необходимости.
- Освобождать batch после завершения всех comparisons для него.
- Ограничивать in-memory prepared-profile cache.
- Не хранить worker pool между независимыми calls.
- Возвращаемый список результатов остается единственным обязательным объектом,
  живущим до завершения public call.
- Измерять RSS и peak size временных arrays.
- Для cache directory считать disk cache persistent, memory cache process-local.

## 9. Тесты

Добавить или сохранить тесты на:

- public `compare_many(metric="co")` backward compatibility;
- каждый metric отдельным invocation;
- internal bounded batching с target list больше batch size;
- target order и duplicate targets через несколько batches;
- prepared и unprepared targets;
- custom model fallback;
- all thresholds and orientations;
- score profiles без scan;
- mixed strand bundles;
- serial vs packed Numba comparison;
- PWM, BaMM, Dimont, Slim и SiteGA batch scan parity;
- mixed model-family dispatcher;
- cache hit/miss и algorithm version;
- in-memory cache hit без disk read;
- LRU eviction и batch cleanup;
- sequence/background fingerprint call count;
- pickle payload validation;
- stable results при разных `NUMBA_NUM_THREADS`.

## 10. Benchmark protocol

Перед каждым изменением отдельно измерять:

- model loading;
- sequence generation/loading;
- preparation context и fingerprints;
- cold preparation;
- warm disk-cache preparation;
- warm in-memory-cache preparation;
- scan;
- normalization;
- anchor collection;
- packed profile creation;
- comparison kernel;
- IPC/process startup, если fallback используется;
- RSS и temporary batch memory.

Использовать два режима:

- cold cache;
- warm cache в том же процессе.

Benchmark-набор:

```text
HOCOMOCO in-vitro/in-vivo models
10 000 sequences x 100 bp
target batches 64, 256 и 512
NUMBA_NUM_THREADS=1, 2, 6
one metric per run
```

Результаты проверять на полном маленьком reference set перед замером
большого набора.

## Порядок внедрения

1. Зафиксировать serial baseline и regression fixtures.
2. Перенести bounded target batching в `compare_many()`.
3. Добавить bounded in-memory cache.
4. Добавить private preparation context для fingerprints.
5. Перевести prepared и unprepared target batches на parent-process packed
   Numba path.
6. Провести A/B benchmark serial, old ProcessPool и Numba batch.
7. Добавить PWM batch scanner.
8. Добавить rolling scanner для BaMM/Dimont/Slim.
9. Добавить SiteGA dispatcher.
10. Оптимизировать batch normalization только после измерения bottleneck.
11. Обновить серверный скрипт до тонкого single-metric caller.

## Критерии завершения

- Target batching полностью находится внутри `mimosa`.
- Серверный скрипт не подготавливает и не shard-ит targets самостоятельно.
- Публичные сигнатуры `compare_many()` и `prepare_profile()` совместимы.
- За один call обрабатывается одна metric.
- Warm in-memory cache не читает disk pickle повторно.
- RSS ограничен размером target batch и bounded memory cache.
- Packed Numba path быстрее serial и текущего ProcessPool на prepared targets.
- Scores, offsets, orientations, n_sites и tie-breaking совпадают с serial path.
- Все model families имеют отдельные regression tests и benchmark.
- Не возникает nested Numba/process oversubscription.
