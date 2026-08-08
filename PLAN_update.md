# План дальнейшей оптимизации mimosa

## Цель

Ускорить сравнение больших коллекций motif-моделей без изменения публичного
контракта `compare_many()` и `prepare_profile()`.

Основной сценарий:

```text
1579 query × 1595 target × 3 metric × 4 min_logerr
10 000 sequences × 100 bp
```

План разделён на две крупные части:

1. Универсальные оптимизации, работающие для любого `PreparedProfile`.
2. Batch scan kernels для встроенных типов моделей.

## Текущее состояние

- Pickle-кеш prepared profiles уже используется как доверенный локальный кеш.
- Warm-cache запуск примерно в 3.5 раза быстрее полного расчёта на benchmark-наборе.
- Серверный скрипт обрабатывает targets bounded-батчами.
- `compare_many()` переиспользует forked worker-pool для prepared target-батча.
- Scan и transform normalization уже имеют Numba parallel kernels внутри одного профиля.
- Обход разных target-моделей и часть comparison orchestration остаются Python-кодом.
- `PreparedProfile.__init__` тратит значительную часть времени подготовки на повторную проверку anchors.

## Ограничения

- Не добавлять новый публичный API для target shards или worker sessions.
- Сохранить `compare_many()` единой публичной точкой сравнения.
- Не использовать nested parallelism: одновременно parallel по targets и по rows нельзя.
- Сохранить порядок targets и существующий tie-breaking.
- Для pickle-кеша явно считать cache directory доверенным.
- Не менять математическую семантику метрик без отдельного согласования.

# Часть I. Универсальные пути

Эта часть применяется одинаково к `PWM`, `BaMM`, `Dimont`, `Slim`, `SiteGA`,
`ScoreProfile` и пользовательским моделям после получения `PreparedProfile`.

## I-0. Baseline и измерения

Перед каждой оптимизацией замерять отдельно:

- чтение моделей;
- генерацию/загрузку sequences;
- cache key и fingerprint;
- scan;
- normalization;
- сбор anchors;
- создание `PreparedProfile`;
- comparison kernel;
- IPC и startup worker-pool;
- RSS и размер временных batch-массивов.

Использовать benchmark с двумя режимами:

- `cache=None`, полный расчёт;
- prepared targets с одним batch-pool.

Принимать изменение только после проверки результатов на полном маленьком
эталонном наборе.

## I-1. Trusted fast path для `PreparedProfile`

### Задача

Убрать повторную проверку уже корректных профилей, созданных непосредственно
внутри `prepare_profile()`.

### Реализация

Добавить приватный путь создания профиля, например:

```text
PreparedProfile._from_validated(...)
```

Использовать его только после внутреннего вызова:

```text
RaggedArray → _fit_normalize → collect_both_anchors
```

Обычный конструктор оставить полностью проверяющим для:

- публичного ручного создания `PreparedProfile`;
- pickle-cache payload;
- будущих внешних источников данных.

### Дополнительная проверка

При сохранении обычной валидации исправить проверку anchors так, чтобы она
проверяла обе границы:

```text
0 <= position < row_length
```

### Критерий

Сократить время подготовки профиля минимум на 30% без изменения результатов.

## I-2. Оптимизация cache key

### Задача

Не пересчитывать один и тот же fingerprint sequences несколько раз.

### Реализация

- При `background is sequences` вычислять `sequence_fingerprint` один раз.
- Не использовать кеш fingerprint по `id()` без гарантии неизменяемости массивов.
- Для длительного сценария рассмотреть объект-контекст с явно вычисленными
  fingerprints, но не добавлять его в публичный API.

### Критерий

Снизить стоимость warm-cache `prepare_profile()` без изменения cache key.

## I-3. Один внутренний batch-вызов для нескольких метрик

### Задача

Сейчас сервер вызывает `compare_many()` отдельно для `co`, `dice` и `cosine`.
Targets уже подготовлены, но query/config и worker-задания передаются трижды.

### Реализация

Сохранить внешний вызов `compare_many()`, но добавить внутренний путь, который:

- принимает набор metric codes внутри одного worker task;
- использует один prepared query;
- проходит target batch один раз на уровне orchestration;
- возвращает отдельные результаты для каждой метрики.

Если расширение `metric` до iterable окажется необходимым, оно должно быть
обратно совместимым со строковым `metric="co"`.

### Критерий

Сохранить идентичность результатов и уменьшить IPC/scheduler overhead.

## I-4. Batch comparison только на Numba threads

### Задача

Убрать `ProcessPool` для уже подготовленного target batch и использовать один
родительский процесс с Numba threads.

### Архитектура

`compare_many()` получает список `PreparedProfile`, затем внутренне строит
packed batch:

- flat score data для forward/reverse;
- offsets профилей и строк;
- flat anchor positions;
- offsets anchors;
- таблицу соответствия batch index → target name.

Python должен только подготовить packed arrays и собрать результаты. Основной
цикл должен быть Numba-kernel:

```text
prange(target_index)
    metric
        orientation
            shift
                score one alignment
```

Публичный API при этом не меняется.

### `_score_shift_best`

Для `min_logerr <= 0` начать с этого kernel: у него нет shared scratch arrays,
и parallel по targets реализуется проще.

### `_score_shift_csr`

Для положительного threshold необходимо сделать scratch state независимым:

- buffers per target; или
- buffers per Numba thread; или
- двухпроходный алгоритм без общего `seen/candidates`.

Не использовать один `seen` и один `candidates` на все `prange` итерации.

### Политика parallelism

- Большие target batches: parallel по targets.
- Малые target batches: возможно parallel по rows.
- Не совмещать оба режима одновременно.
- Число threads выбирать по размеру batch и `NUMBA_NUM_THREADS`.

### Численная корректность

`prange` может менять порядок float-редукции. Проверить:

- scores через `assert_allclose`;
- exact совпадение offset/orientation/n_sites;
- tie-breaking на равных scores;
- стабильность при разных `NUMBA_NUM_THREADS`.

### Критерий

В режиме prepared targets получить ускорение comparison минимум 2 раза против
текущего serial пути без ухудшения качества результатов.

## I-5. Memory и lifecycle

- Сохранять bounded target batches.
- Не создавать копии packed arrays без необходимости.
- Освобождать batch после обработки всех query.
- Не держать весь набор targets в RAM на машинах с недостаточной памятью.
- Оставить pickle-диск кеш для cold start и восстановления после остановки.

## I-6. Проверки части I

Добавить/сохранить тесты на:

- все три метрики;
- target order;
- duplicate targets;
- `min_logerr <= 0` и положительные thresholds;
- все четыре orientations;
- score profiles без scan;
- mixed strand bundles;
- сравнение serial, ProcessPool и Numba batch paths;
- pickle cache hit/miss и смену cache algorithm version.

# Часть II. Batch scan kernels для моделей

Эта часть ускоряет именно получение raw score profiles. После подготовки все
модели используют универсальную часть I.

## II-0. Общий private contract

Добавить внутренний, не экспортируемый путь batch scan:

```text
models + EncodedSequences
    → packed forward/reverse raw score profiles
```

Требования:

- сохранить исходный порядок моделей;
- поддерживать ragged offsets;
- не смешивать score profiles разных моделей;
- вернуть metadata геометрии для последующей normalization;
- не менять `MotifModel.scan_into()` public contract.

Batch scan должен выбирать kernel по типу и геометрии моделей до запуска Numba.
Не вызывать уже parallel kernel отдельно для каждой модели внутри outer
`prange`, чтобы не получить nested parallelism.

## II-1. PWM batch scanner

### Вход

- packed PWM weights;
- одинаковая или padded motif length;
- sequence data/offsets;
- массив motif lengths.

### Реализация

Добавить kernels для:

- forward scan;
- reverse scan;
- обеих strands;
- `prange(model_index)` либо `prange(model_index, sequence_index)`.

Модели разных длин можно обрабатывать:

- группами по motif length; или
- padded weights с явным массивом lengths.

### Проверки

- serial PWM scan == batch PWM scan;
- N bases;
- forward/reverse;
- модели длины 1 и максимальной длины.

## II-2. Batch rolling scanner для BaMM, Dimont и Slim

Эти модели имеют общую rolling-геометрию и отличаются параметрами `order` и
`motif_length`.

### Реализация

- группировать модели по `(order, motif_length)`;
- передавать packed weights группы в один rolling kernel;
- сохранить отдельные forward/reverse paths;
- не разрешать модели с несовместимой геометрией в одной группе.

### Проверки

- order `0`;
- максимальный поддерживаемый order;
- разные motif lengths;
- сравнение с текущим `batch_rolling_forward/reverse` для каждой модели.

## II-3. Batch rolling scanner для SiteGA

SiteGA использует rolling kernel с фиксированным `kmer=2` и другой связью
между `motif_length` и числом score positions.

### Реализация

- отдельный SiteGA batch dispatch;
- не объединять его без проверки с BaMM/Dimont/Slim;
- сохранить существующие размеры и coordinate contract.

### Проверки

- минимальная длина SiteGA;
- forward/reverse;
- совпадение offsets и scores с текущим scanner.

## II-4. Пользовательские `MotifModel`

Автоматически распараллелить произвольный `scan_into()` нельзя: это Python
callback с неизвестной потокобезопасностью и стоимостью.

Для таких моделей оставить fallback:

- текущий последовательный `scan_into()`;
- универсальный batch comparison из части I после получения profiles.

Опциональный будущий extension point добавлять только как отдельный явно
заявленный capability, например `batch_scan_into`, но не менять обязательный
public contract сейчас.

## II-5. Batch normalization и profile creation

После batch scan можно оптимизировать подготовку двумя этапами:

1. Сначала реализовать batch scan без изменения normalization.
2. Затем при необходимости добавить parallel normalization по model index.

`PreparedProfile.__init__` для внутренних результатов должен использовать
trusted fast path из части I.

Не объединять таблицы normalization разных моделей: каждая модель получает
собственную calibration distribution и собственные anchors.

## II-6. Проверки части II

Для каждого семейства сравнить:

- scores;
- offsets;
- forward/reverse identity;
- normalized values;
- anchors;
- итоговые comparison results.

Проверить также смешанные коллекции, например:

```text
PWM + BaMM + SiteGA + Dimont + Slim
```

В смешанном batch dispatcher должен разбить модели на совместимые группы,
запустить семейные kernels и восстановить исходный порядок.

# Порядок внедрения

1. Baseline benchmark и численные regression tests.
2. Trusted `PreparedProfile` fast path.
3. Исправление повторного sequence fingerprint.
4. Fused multi-metric path внутри `compare_many()`.
5. Universal packed Numba comparison без ProcessPool.
6. A/B benchmark ProcessPool против Numba-only backend.
7. PWM batch scanner.
8. Общий rolling batch scanner для BaMM/Dimont/Slim.
9. SiteGA batch scanner.
10. Parallel normalization по model batch, только если scan остаётся bottleneck.
11. Поддержка custom batch capability только при реальной потребности.

# Критерии завершения

- Публичные сигнатуры `compare_many()` и `prepare_profile()` совместимы.
- Результаты всех model families совпадают с текущим serial path с заданной
  численной погрешностью.
- Не возникает nested Numba/process oversubscription.
- RSS ограничен размером target batch.
- Для prepared batch Numba-only backend быстрее текущего forked pool.
- Для каждого model family есть отдельный benchmark и regression test.
