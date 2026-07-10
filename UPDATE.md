# План оптимизации MIMOSA

## Статус и область изменений

Документ описывает потенциальное обновление вычислительного ядра MIMOSA без изменения математической семантики
сравнения мотивов. Основные цели:

- убрать создание больших временных матриц окон в `profile`-режиме;
- объединить сбор, дедупликацию и оценку кандидатов в последовательный поток вычислений;
- использовать Numba как единственный механизм CPU-параллелизма;
- удалить `joblib.Parallel`, `joblib.delayed` и backend `loky` из выполнения сравнений;
- сохранить читаемый Python orchestration layer и небольшие специализированные численные ядра;
- не ухудшить поведение на малых входах и не изменить результаты существующих сравнений.

Обновление не предполагает перенос проекта на другой язык или добавление Cython, C++ либо Rust. Возможность нативного
расширения следует рассматривать только после выполнения и измерения описанных ниже изменений.

## Обязательные ограничения

### Единственный механизм параллелизма

Numba является единственным механизмом распараллеливания вычислений. После обновления не должно быть двух вложенных
или конкурирующих уровней параллелизма.

- В численных ядрах разрешены `parallel=True` и `numba.prange`.
- Список target-моделей всегда обходится последовательно.
- `joblib.Parallel`, `joblib.delayed` и `loky` удаляются из `mimosa.comparison.runner`.
- Параметр `n_jobs` временно сохраняется ради обратной совместимости, но интерпретируется как число потоков Numba.
- CLI-параметр `--jobs` также управляет числом потоков Numba, а не числом процессов или target-задач.
- `--jobs 1` гарантирует последовательное выполнение численных ядер.
- `--jobs -1` использует доступный Numba thread budget.
- На уровне одного публичного вызова число потоков устанавливается один раз и восстанавливается после завершения.

Разрешение thread budget должно быть явным и покрытым тестами:

| Значение | Поведение |
| --- | --- |
| `n_jobs=None` | Не изменять текущий Numba thread mask и учитывать `NUMBA_NUM_THREADS` |
| `n_jobs=1` | Установить один поток и использовать serial entry points |
| `n_jobs=-1` | Использовать максимум потоков, доступный текущему Numba runtime |
| `n_jobs=N`, `N > 1` | Использовать ровно `N` потоков, если `N` не превышает доступный максимум |

Недопустимые значения должны отклоняться существующей валидацией до изменения thread mask. Положительное значение,
превышающее доступный Numba maximum, должно приводить к понятной `ValueError`, а не молча ограничиваться.

Зависимость `joblib` пока остается в `pyproject.toml`, поскольку она используется как формат сериализации моделей и
null distributions. Удаляется только применение `joblib` для параллельного выполнения.

### Имена новых файлов

Новые файлы с Python-кодом не должны содержать символ `_` в имени.

Допустимые примеры:

- `src/mimosa/functions/profilealignment.py`;
- `benchmarks/profilealignment.py`;
- `benchmarks/threadscaling.py`.

Недопустимые примеры:

- `profile_alignment.py`;
- `thread_scaling.py`;
- `benchmark_profile.py`.

Правило относится только к именам новых файлов. Внутри Python-кода сохраняется стандартный `snake_case`. Новые тесты
следует добавлять в существующие `tests/unit_comparison.py`, `tests/unit_functions.py` и `tests/test_integration.py`,
чтобы не создавать новые test-файлы с `_` в имени.

## Исходное состояние

Текущая последовательность вычислений для одного shift выглядит следующим образом:

1. `_collect_model1_window_candidates` создает массивы кандидатов query-модели.
2. `_collect_model2_window_candidates` создает массивы кандидатов target-модели и выполняет local realignment.
3. `_merge_window_candidates` объединяет шесть массивов, вызывает `np.lexsort` и удаляет дубликаты.
4. `_extract_selected_windows` дважды выполняет advanced indexing и создает плотные матрицы окон.
5. `_score_window_collection` передает созданные матрицы отдельному ядру метрики.
6. Операции повторяются для каждого shift и каждой из четырех ориентаций.

Профилирование показало, что при большом числе anchors основное время тратится на merge, сортировку, advanced indexing
и перемещение памяти. Формула метрики занимает существенно меньшую долю времени. Поэтому первое обновление должно
уменьшать число аллокаций, а не добавлять потоки к существующему allocation-heavy алгоритму.

## Целевая архитектура

После обновления вычислительный путь должен состоять из трех четких уровней.

### Python orchestration

Python-код отвечает за:

- валидацию конфигурации;
- подготовку contiguous-массивов с фиксированными dtype;
- подготовку anchors;
- выбор serial или parallel Numba entry point;
- перебор четырех ориентаций;
- формирование `ComparisonResult`;
- обработку progress bar и исключений.

На этом уровне не должно быть циклов по каждому nucleotide position или каждому элементу окна.

### Alignment wrapper

Небольшой wrapper в `alignment.py` отвечает за:

- преобразование имени метрики в внутренний integer code;
- выбор pooled или rowwise семантики;
- подготовку workspace;
- вызов соответствующего Numba-ядра;
- преобразование результата ядра в `(score, shift, n_sites)`.

Numba-ядрам не следует передавать строки, dataclass-объекты, словари или `ComparatorConfig`.

### Numerical kernels

Численные ядра получают только NumPy-массивы и скаляры. Они отвечают за:

- проверку границ окна;
- local realignment target anchors;
- дедупликацию кандидатов;
- последовательный доступ к значениям окон;
- накопление метрики;
- reduction результатов строк;
- поиск лучшего shift с сохранением текущих tie-breaking rules.

Сложные действия следует делить на небольшие helpers с предметными именами. Один helper должен выполнять одну
операцию, а не весь comparison workflow.

## Этап 0. Зафиксировать корректность и baseline

До изменения production-кода нужно получить воспроизводимую точку сравнения.

### Benchmark-сценарии

Создать каталог `benchmarks` без превращения его в Python package и добавить:

- `benchmarks/alignment.py` для isolated profile scorer;
- `benchmarks/thread_scaling.py` для проверки масштабирования Numba;
- при необходимости `benchmarks/target_loop.py` для one-to-many сценария.

Каждый benchmark должен поддерживать фиксированный seed и печатать машиночитаемую TSV-таблицу. Минимальная матрица
входов:

| Класс | Rows | Width | Targets | Назначение |
| --- | ---: | ---: | ---: | --- |
| small | 100 | 100 | 1 | Проверка overhead |
| medium | 1000 | 100 | 32 | Типичный one-to-many |
| large | 4000 | 100 | 32 | Нагрузка на scorer |
| collection | 1000 | 100 | 1000 | Массовое сравнение |

Для каждого класса измерять:

- best-anchor mode;
- threshold-anchor mode с несколькими реалистичными порогами;
- `co`, `dice`, `co_rowwise`, `dice_rowwise` и `cosine`;
- одну и четыре ориентации;
- cold run отдельно от прогретого run;
- wall time;
- peak RSS в отдельном процессе;
- число selected sites;
- итоговый score, shift и orientation.

### Reference implementation

Текущий алгоритм следует сохранить только как test helper. Production-код не должен постоянно содержать две полные
реализации одного алгоритма.

В существующие тестовые модули добавить differential tests, которые сравнивают reference и optimized paths на:

- пустых profile bundles;
- строках нулевой длины;
- ragged profiles;
- окнах радиуса 0 и больше 0;
- отрицательных, нулевых и положительных shifts;
- best-anchor и threshold-anchor режимах;
- дубликатах, одновременно созданных query и target anchors;
- target anchors, которые после realignment попадают в один query position;
- всех четырех ориентациях;
- всех поддерживаемых метриках;
- tie по score и tie по `n_sites`;
- случайных малых массивах для нескольких фиксированных seeds.

Критерии эквивалентности:

- `shift`, `orientation` и `n_sites` совпадают точно;
- score сравнивается через `np.testing.assert_allclose` с `rtol=1e-6` и `atol=1e-7`;
- порядок результатов one-to-many совпадает с порядком входных targets.

## Этап 1. Streaming scoring без изменения candidate selection

Это изменение должно быть минимальным и низкорисковым.

### Реализация

Добавить `src/mimosa/functions/alignment.py`. На первом этапе модуль содержит только kernels, которые получают
готовые `rows`, `pos1`, `pos2` и непосредственно читают исходные score matrices.

Текущая последовательность:

```text
candidates -> dense windows1/windows2 -> metric kernel
```

заменяется на:

```text
candidates -> streaming metric kernel
```

Для pooled `co` и `dice` ядро за один проход накапливает:

- сумму значений query;
- сумму значений target;
- сумму поэлементных минимумов.

Для `co_rowwise` и `dice_rowwise` ядро вычисляет score каждого окна, сразу добавляет finite score к общей сумме и
увеличивает счетчик. Для `cosine` аналогично накапливаются dot product и две нормы одного окна.

### Требования к читаемости

- Python wrapper выполняет dispatch по имени метрики.
- Внутреннее ядро получает integer mode или boolean flags, но не строку.
- Pooled overlap, rowwise overlap и rowwise cosine остаются отдельными функциями.
- Общая проверка границ и проход по окну выносятся только тогда, когда это не требует сложной абстракции.
- Не использовать Numba-specific tricks вроде `literally`, generated overloads или динамического dispatch без измеримой
  необходимости.

### Что пока не менять

- сбор query candidates;
- realignment target candidates;
- `np.lexsort` и текущую дедупликацию;
- Python-цикл по shifts;
- выбор ориентации;
- параллелизм.

### Критерий принятия

- Все differential tests проходят.
- Peak RSS threshold-heavy сценария уменьшается не менее чем на 30%.
- Threshold-heavy scorer ускоряется не менее чем в 1.5 раза.
- Best-anchor режим не замедляется более чем на 5%.

Если критерии не выполнены, этап не усложняется автоматически: сначала повторно профилируется новый путь.

## Этап 2. Компактная дедупликация кандидатов

После устранения dense window matrices следующим bottleneck, вероятно, станет `_merge_window_candidates`.

### Упрощение ключа

Для одного фиксированного shift выполняется:

```text
pos2 = pos1 + shift
```

Следовательно, уникальность triplet `(row, pos1, pos2)` эквивалентна уникальности пары `(row, pos1)`. Нет необходимости
сортировать три отдельных массива.

### Вариант A: компактные integer keys

Сначала реализовать наиболее простой вариант:

```text
key = row * profile_width + pos1
```

Порядок действий:

1. Создать один `int64` массив достаточного размера.
2. Добавить валидные query keys.
3. Добавить realigned target keys.
4. Отсортировать заполненную часть массива.
5. За один линейный проход пропустить повторяющиеся keys.
6. Восстановить `row` и `pos1` через division и remainder.
7. Передать уникальные позиции streaming scorer.

Перед использованием кодирования нужно проверить отсутствие переполнения для допустимых shapes. Эта проверка остается
в Python wrapper и не выполняется во внутреннем цикле.

### Вариант B: reusable marks workspace

Переходить к workspace следует только если профилирование показывает, что сортировка остается значимой. Workspace
имеет форму `(n_rows, profile_width)` и dtype `int32`. Для каждого shift увеличивается generation counter:

```text
marks[row, pos1] == generation
```

Это позволяет отмечать кандидаты без сортировки и без очистки всего массива после каждого shift. При переполнении
generation counter workspace очищается явно; этот редкий случай должен быть покрыт unit test.

### Выбор варианта

Нельзя оставлять оба production paths без необходимости. После benchmark выбирается один основной алгоритм:

- integer keys, если он достаточно быстр и лучше работает на sparse anchors;
- marks workspace, если сортировка доминирует и дополнительная память приемлема.

Если sparse и dense режимы действительно требуют разных алгоритмов, порог выбора должен быть получен из benchmark,
оформлен именованной константой и объяснен коротким комментарием.

### Критерий принятия

- `n_sites` полностью совпадает с reference implementation.
- Tie-breaking не меняется.
- Новый merge быстрее текущего минимум на 20% в сценарии, где merge занимает не менее 20% общего времени.
- Для small/best-anchor сценария нет регрессии более 5%.

## Этап 3. Fused kernel для одного shift

После стабилизации streaming scoring и deduplication можно убрать промежуточные candidate arrays.

### Контракт ядра

Основной serial kernel одного shift получает:

- query и target score matrices;
- lengths;
- query anchor positions и offsets по строкам;
- target anchor positions и offsets по строкам;
- shift;
- window radius;
- realign window;
- metric code;
- reusable workspace.

Он возвращает:

```text
(score, n_sites)
```

### Внутренний поток

Для каждой строки ядро выполняет:

1. Отмечает query anchors, для которых оба окна полностью входят в profiles.
2. Для каждого target anchor вычисляет ожидаемую query position.
3. Ищет максимум query score в пределах `realign_window`.
4. Проверяет границы выровненных окон.
5. Пропускает уже отмеченный `(row, pos1)`.
6. Непосредственно накапливает выбранную метрику.
7. Возвращает row-local partial result.

Flat arrays anchors следует преобразовать в CSR-подобное представление:

- `positions` содержит позиции anchors;
- `offsets` длины `n_rows + 1` определяет диапазон anchors каждой строки.

Такой формат позволяет обрабатывать одну строку независимо и позднее безопасно добавить `prange`.

### Разделение функций

Чтобы fused kernel оставался читаемым, выделить небольшие helpers:

- проверка границ окна;
- поиск realigned query position;
- отметка query candidates;
- отметка target candidates;
- pooled overlap accumulation;
- rowwise overlap accumulation;
- cosine accumulation;
- сравнение двух shift results с текущими tie-breaking rules.

Helpers должны иметь предметные имена. Не следует создавать универсальный helper с большим числом flags, если три
коротких специализированных функции читаются лучше.

### Граница fusion

На этом этапе четыре ориентации остаются на Python-уровне. Цикл по shifts также первоначально остается Python-кодом.
Его перенос внутрь Numba допустим только если profiling показывает, что вызовы одного-shift kernel занимают более
10-15% общего времени после устранения аллокаций.

### Критерий принятия

- Threshold-heavy scorer ускоряется минимум в 2 раза относительно исходного baseline.
- Peak RSS уменьшается минимум на 50% относительно исходного baseline.
- Best-anchor режим не замедляется более чем на 5%.
- Размер и сложность orchestration-функций в `comparison/profile.py` уменьшаются.

## Этап 4. Numba parallelism

Параллелизм добавляется только после оптимизации serial algorithm. Иначе потоки будут параллельно выполнять лишние
аллокации и сортировки.

Production-включение `parallel=True` имеет обязательное предварительное условие: изменения этапа 5 по удалению
`joblib`-параллелизма уже применены либо входят в тот же атомарный commit. Изолированные прототипы `prange` разрешены в
benchmark scripts, но в основной ветке не должно существовать состояния, где одновременно активны `loky` и parallel
Numba.

### Общая модель потоков

Число потоков устанавливается один раз на границе публичного comparison call:

1. Сохранить `numba.get_num_threads()`.
2. Преобразовать `n_jobs` в допустимое число потоков.
3. Вызвать `numba.set_num_threads()`.
4. Последовательно выполнить все target comparisons.
5. В `finally` восстановить предыдущее значение.

Context manager следует разместить в существующем `comparison/runner.py`, если он остается небольшим. Если потребуется
отдельный модуль, его имя должно быть без `_`, например `threadcontrol.py`.

### Кандидаты для `prange`

Проверять параллелизацию в следующем порядке.

#### Profile alignment

Первый приоритет — строки fused alignment kernel. Строки независимы и имеют естественную reduction-модель.

- Внешний цикл выполняется как `prange(n_rows)`.
- Каждый поток пишет только в row-local workspace и row-local partial results.
- После parallel region выполняется детерминированная reduction.
- Ни один поток не обновляет общий scalar или общий участок marks workspace.

#### Tail normalization

Второй приоритет — внешний цикл по profile rows в `apply_score_log_tail_table_to_profile_bundle`. Каждая строка
нормализуется независимо.

#### Sequence scanning

Третий приоритет — внешний цикл `row_index` в forward, reverse и both-strands scanning kernels. Параллельное both-strands
ядро предпочтительнее двух отдельных parallel regions.

### Serial и parallel entry points

Для малых входов overhead `prange` может превышать полезную работу. Поэтому нужны два entry points, использующие общие
маленькие helpers:

- serial entry point с `range`;
- parallel entry point с `prange` и `parallel=True`.

Python wrapper выбирает entry point по двум условиям:

- запрошено больше одного потока;
- размер работы выше benchmark-derived threshold.

Threshold оформляется именованной константой рядом с wrapper. Формула work size должна быть простой, например
`n_rows * profile_width * number_of_shifts`, и сопровождаться объяснением происхождения порога.

### Детерминированность

Parallel reduction может изменить порядок floating-point сложения. Требования:

- orientation, shift и `n_sites` остаются полностью детерминированными;
- score укладывается в установленный tolerance;
- повторные вызовы с одинаковыми входами дают одинаковый результат в рамках выбранной reduction scheme;
- `fastmath=True` не добавляется в новые kernels автоматически: решение принимается отдельным benchmark и тестом
  численной устойчивости.

### Диагностика Numba

Перед принятием parallel kernel проверить `parallel_diagnostics(level=4)`:

- требуемый внешний цикл действительно parallel;
- вложенные циклы сериализованы ожидаемым образом;
- нет неожиданной race condition;
- allocations не выполняются внутри каждой итерации `prange`, если их можно подготовить заранее.

### Критерий принятия

На четырех физических ядрах:

- large alignment ускоряется минимум в 1.5 раза относительно optimized serial kernel;
- tail normalization либо scanning ускоряется минимум в 1.3 раза, иначе соответствующий kernel остается serial;
- small workload автоматически использует serial path;
- small workload не замедляется более чем на 5%;
- память не растет более чем на 25% относительно optimized serial path.

Если конкретный kernel не достигает критерия, наличие `prange` само по себе не является причиной оставлять его.

## Этап 5. Удаление joblib-параллелизма

Этот этап завершает переход к единственной модели параллелизма.

С точки зрения порядка production-изменений этот этап является prerequisite для этапа 4. Нумерация отражает области
работы, а не разрешает сначала включить Numba threads поверх существующего `loky`. Допустимы два безопасных варианта:

- сначала перейти на последовательный target loop, а следующим commit включить parallel Numba;
- удалить `loky` и включить parallel Numba одним атомарным commit после завершения isolated benchmarks.

### Изменения runner

В `src/mimosa/comparison/runner.py`:

- удалить импорт `Parallel` и `delayed`;
- удалить backend `loky`;
- удалить process-specific ветвление;
- заменить `_run_target_comparisons` на простой последовательный обход targets;
- сохранить progress reporting и порядок результатов;
- сохранить очистку target-local runtime cache в вызывающих стратегиях;
- оборачивать весь вызов в thread-control context manager, а не менять число потоков для каждого target.

Целевой поток one-to-many:

```text
resolve Numba threads
prepare query once
for target in targets:
    prepare target
    run Numba kernels using configured threads
    build result
restore previous Numba thread count
```

### Семантика конфигурации

На первом релизе сохраняются существующие имена `n_jobs` и `--jobs`, чтобы не ломать Python API и CLI scripts.
Документация меняется следующим образом:

- раньше: число target-level joblib workers;
- после обновления: максимальное число потоков Numba для одного comparison kernel.

В следующем major release можно отдельно рассмотреть rename `n_jobs` в `num_threads`, но смешивать это API-изменение с
оптимизацией kernels не следует.

### Direct motif mode

Прямое motif-сравнение пока остается последовательным, поскольку отдельные matrix alignments слишком малы для
эффективного process-level parallelism. Если массовые direct comparisons станут bottleneck, допустим только Numba
batch kernel с `prange` по targets. Возвращать `joblib` или добавлять второй thread pool нельзя.

Batch kernel для motif mode является отдельным будущим этапом и принимается только при наличии benchmark для больших
коллекций. Для variable motif lengths потребуется padded representation плюс массив lengths; добавлять эту сложность
без измеримой необходимости не следует.

### Joblib serialization

Следующие применения `joblib` не относятся к параллелизму и сохраняются:

- чтение legacy serialized models;
- запись совместимых serialized models;
- чтение и запись `.joblib` null distributions.

README должен ясно разделять joblib как формат сериализации и Numba как механизм параллельных вычислений.

### Критерий принятия

- В comparison execution path отсутствуют `Parallel`, `delayed` и `loky`.
- One-to-many возвращает результаты в прежнем порядке.
- Исключения target comparison не скрываются и не меняют тип.
- Progress bar корректно работает в последовательном target loop.
- `--jobs 1`, `--jobs 2` и `--jobs -1` проверены integration tests.
- После публичного вызова прежнее число потоков Numba восстановлено даже при исключении.

## Этап 6. Очистка API и документации

После стабилизации производительности:

- обновить help для `--jobs`;
- обновить раздел performance в README;
- явно описать, что targets обрабатываются последовательно;
- описать критерий выбора serial/parallel kernel;
- исправить существующее упоминание joblib threads в docstring runner;
- удалить устаревшие process-level tests и заменить их thread-control tests;
- проверить, что новые внутренние функции не попали в публичный `mimosa.__all__` без необходимости;
- не добавлять пользовательские performance flags, пока автоматический выбор работает предсказуемо.

## План тестирования

### Unit tests

Добавить или расширить тесты для:

- streaming metrics против текущих формул;
- compact key encoding и decoding;
- deduplication query/target overlap;
- multiple target anchors после realignment;
- CSR anchor offsets;
- serial и parallel kernels;
- thread count resolution;
- восстановления thread count после успешного вызова;
- восстановления thread count после исключения;
- последовательного one-to-many порядка;
- progress iterator без joblib;
- полного набора tie-breaking rules.

### Integration tests

Проверить существующие CLI-сценарии для:

- `profile` с PWM/PWM;
- `profile` с разными типами моделей;
- precomputed scores;
- threshold selection;
- `motif` direct comparison;
- one-to-many API;
- `build-null`;
- `--jobs 1`, `--jobs 2`, `--jobs -1`.

Integration tests не должны проверять конкретное ускорение, поскольку CI hardware нестабилен. Performance gates
проверяются отдельными benchmark scripts.

### Обязательные команды проверки

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python benchmarks/profilealignment.py
uv run python benchmarks/threadscaling.py
```

## Performance gates

Изменение считается успешным только при выполнении всех условий:

| Показатель | Требование |
| --- | --- |
| Корректность | Все unit и integration tests проходят |
| Численная эквивалентность | Score в пределах `rtol=1e-6`, `atol=1e-7` |
| Структурная эквивалентность | Точное совпадение shift, orientation и n_sites |
| Threshold-heavy serial | Не менее 2x быстрее исходного baseline |
| Best-anchor serial | Регрессия не более 5% |
| Parallel large | Не менее 1.5x быстрее optimized serial на 4 физических ядрах |
| Parallel small | Автоматически выбран serial path |
| Peak RSS | Снижение не менее 50% в threshold-heavy benchmark |
| One-to-many | Нет process startup и serialization overhead |

Если parallel kernel не масштабируется из-за memory bandwidth, остается optimized serial kernel. Не следует сохранять
параллельную реализацию только ради наличия многопоточности.

## Риски и способы контроля

### Слишком крупное Numba-ядро

Риск: fusion может создать длинную трудно тестируемую функцию.

Контроль:

- разделять boundary checks, realignment, candidate marking и metric accumulation;
- сохранять Python dispatch;
- не переносить подготовку моделей и выбор ориентаций в Numba;
- ограничивать helpers одной предметной операцией.

### Регрессия sparse best-anchor mode

Риск: dense marks workspace или полный scan всей строки могут быть дороже небольшого списка anchors.

Контроль:

- сначала реализовать compact integer keys;
- добавлять dense workspace только после profiling;
- не вводить две ветви без подтвержденного crossover point.

### Изменение floating-point результата

Риск: parallel reduction меняет порядок сложения.

Контроль:

- фиксированная row-local accumulation;
- последовательная reduction row partials, если ее стоимость мала;
- differential tests на всех метриках;
- отдельное обоснование `fastmath`.

### Oversubscription

Риск устраняется архитектурно: target loop последовательный, а единственный thread pool принадлежит Numba.

### Холодный JIT start

Для новых стабильных kernels следует отдельно проверить `cache=True`. Это изменение принимается только после проверки
корректной invalidation и отсутствия проблем с package installation. JIT cache не заменяет profile data cache и не
должен смешиваться с ним в документации.

## Последовательность реализации

Рекомендуемые независимые изменения и commits:

1. Добавить benchmarks и reference differential tests.
2. Добавить streaming metric kernels без изменения candidates.
3. Перейти на компактную дедупликацию `(row, pos1)`.
4. Добавить CSR anchors и fused serial kernel одного shift.
5. Перенести цикл shifts в Numba только при подтвержденном Python overhead.
6. Добавить единый Numba thread-control scope без включения parallel kernels.
7. Удалить joblib parallel execution и сделать target loop последовательным.
8. Добавить serial/parallel entry points для profile alignment.
9. Проверить и при необходимости распараллелить tail normalization.
10. Проверить и при необходимости распараллелить both-strands scanning.
11. Обновить CLI help, README и integration tests.
12. Выполнить полную correctness и performance verification.

Каждый commit должен проходить полный test suite. Performance-related commit должен включать таблицу benchmark до и
после изменения.

## Definition of Done

- [ ] Python API и CLI сохраняют совместимость результатов.
- [ ] Новые файлы с кодом не содержат `_` в имени.
- [ ] `joblib` не используется для параллельного выполнения.
- [ ] Numba является единственным механизмом CPU-параллелизма.
- [ ] Target collections обрабатываются последовательным циклом.
- [ ] Число потоков устанавливается один раз на публичный вызов и всегда восстанавливается.
- [ ] Dense selected-window matrices больше не создаются.
- [ ] Candidate deduplication использует пару `(row, pos1)`.
- [ ] Serial implementation оптимизирована до добавления `prange`.
- [ ] Small workloads используют serial path.
- [ ] Parallel kernels проходят Numba diagnostics и scaling gates.
- [ ] Старый алгоритм отсутствует в production-коде и сохраняется только как test reference.
- [ ] Все тесты, Ruff checks и benchmark gates проходят.
- [ ] README различает joblib serialization и Numba parallel execution.
