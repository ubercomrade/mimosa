# План оптимизации profile pipeline в Python и Julia

## 1. Цель

Ускорить точный (`exact`) profile pipeline в Python и Julia для основного
workload:

- 10 000 последовательностей длиной 100 bp;
- один query PWM против 50 target PWM;
- profile metric `co`;
- `search_range=10`, `window_radius=5`, `realign_window=3`;
- serial и four-thread execution.

Текущий измеренный baseline на Intel Core i7-11370H:

| Реализация | 1 поток | 4 потока |
|---|---:|---:|
| Python | 25.552 s | 21.704 s |
| Julia | 18.594 s | 17.161 s |

Основные цели: устранить повторную работу в empirical normalization, добавить
параллелизм верхнего уровня по targets и сократить накладные расходы alignment,
не изменяя научную семантику.

## 2. Обязательные ограничения

1. Python остается источником научной семантики и oracle behavior.
2. Не менять profile metrics, anchor selection, offsets, orientation priority,
   site counts и tie-breaking.
3. Не менять порядок `Float64` accumulation внутри одного
   `(orientation, shift)`.
4. Не использовать approximate quantiles, histogram normalization, `fastmath`
   или новые runtime dependencies.
5. Не регенерировать frozen compatibility fixtures.
6. Для 1-vs-many параллелить targets как верхний независимый уровень.
7. Внутри параллельного target worker использовать serial scan, normalization
   и alignment. Не допускать nested parallelism.
8. Результаты должны сохранять исходный порядок targets. Worker exception
   должен прерывать операцию, не возвращая частично заполненный результат.
9. Перед редактированием выполнить `git status --short` и сохранить все
   посторонние пользовательские изменения.
10. Каждое изменение принимать только после same-machine benchmark и
    compatibility tests.

## 3. Этап 1: разделить benchmark по стадиям

Расширить:

- `Mimosa.jl/benchmark/cross_language_profile.py`;
- `Mimosa.jl/benchmark/cross_language_profile.jl`.

Измерять отдельно:

1. query scan;
2. query normalization;
3. target scan;
4. target normalization;
5. anchor collection;
6. prepared-to-prepared alignment 1-vs-1;
7. prepared 1-vs-50;
8. end-to-end 1-vs-50.

Использовать один общий FASTA и один общий MEME для обеих реализаций. Исключить
I/O и JIT из timed region. Выполнять warm-up и не менее трех repetitions.
Записывать median, minimum, runtime versions, CPU, thread policy и allocations
для Julia. Для Python при необходимости измерять peak RSS в отдельном
subprocess, поскольку `tracemalloc` не учитывает всю native memory.

До начала оптимизаций сохранить baseline JSON. После каждого этапа повторять
те же измерения без изменения входов и параметров.

## 4. Этап 2: fused empirical normalization

### Python

Основные файлы:

- `src/mimosa/functions/tails.py`;
- `src/mimosa/comparison/profile.py`;
- `tests/unit_functions.py`;
- `tests/unit_comparison.py`.

### Julia

Основные файлы:

- `Mimosa.jl/src/profiles/normalization.jl`;
- `Mimosa.jl/src/profiles/alignment.jl`;
- `Mimosa.jl/src/comparison/profile_comparison.jl`;
- `Mimosa.jl/test/unit/test_profiles.jl`;
- `Mimosa.jl/test/properties/test_properties.jl`;
- `Mimosa.jl/test/compatibility/test_profile_fixtures.jl`.

### Алгоритм

Для случая, когда calibration scores совпадают с нормализуемым profile bundle:

1. Собрать valid `Float32` scores двух strands в один плоский массив.
2. Выполнить один descending `argsort`/`sortperm`.
3. Одним последовательным проходом определить группы равных scores.
4. Для группы с последним sorted index `j` вычислить
   `Float32(-log10(Float64(j) / Float64(total)))` с учетом индексации языка.
5. Сразу scatter-ить значение в normalized output в исходном порядке.
6. Одновременно построить lookup table из unique scores и log-tail values.
7. Не выполнять binary search для каждого исходного score.

Python должен заменить лишнюю комбинацию `np.sort` и `np.unique` одним sort и
run-length проходом. Julia должна обобщить существующий
`_fit_transform_empirical` на двухstrandовый model-derived bundle.

Для symmetric `ScoreProfile` нормализовать данные один раз и сохранить alias:
`forward === reverse` в Julia и одно разделяемое immutable представление в
Python, если это совместимо с текущим batch contract.

При отдельном background оставить существующий table lookup path. Sorted-merge
transform можно рассматривать только как отдельную измеренную оптимизацию.

### Тесты и критерии

Проверить empty, singleton, all-equal, duplicate-heavy, `-0.0f0`, sorted,
reverse-sorted и random finite Float32 inputs, включая ragged empty rows.

Новый результат должен быть exact-equivalent старому `fit + apply` для finite
Float32. Offsets и row order должны сохраняться.

Критерии приема:

- common-case normalization не медленнее `0.45x` старого времени;
- allocations и peak memory ниже baseline;
- frozen fixtures проходят без перегенерации.

## 5. Этап 3: target-level parallelism

### Python

Основные файлы:

- `src/mimosa/comparison/runner.py`;
- `src/mimosa/comparison/profile.py`;
- `tests/unit_comparison.py`;
- `tests/test_integration.py`.

### Julia

Основные файлы:

- `Mimosa.jl/src/profiles/alignment.jl`;
- `Mimosa.jl/src/comparison/profile_comparison.jl`;
- `Mimosa.jl/test/unit/test_parallel.jl`;
- `Mimosa.jl/test/unit/test_profiles.jl`.

Добавить Julia one-to-many методы для model targets:

```julia
compare(query_model, target_models, sequences; execution=...)
compare(prepared_query, target_models, sequences; execution=...)
```

Query готовить один раз. Targets распределять bounded dynamic queue в Julia и
bounded worker pool в Python. Каждый worker должен иметь собственные scan,
normalization, anchor и alignment buffers и записывать результат только в свой
заранее выделенный индекс.

При outer target parallelism:

- Julia worker использует `SerialExecution()` внутри;
- Python worker отключает inner Numba `prange` и использует thread mask 1;
- query bundle и query anchors разделяются только как immutable данные;
- target caches и mutable workspaces не разделяются.

Для single-target API можно сохранить внутренний row-parallel alignment, если
benchmark подтверждает его пользу.

Добавить тесты на empty/singleton targets, ragged imbalance, worker caps,
stable order, exception propagation, nested fallback и serial/threaded
equivalence.

Критерий приема: four-thread 1-vs-50 должен ускоряться не менее чем в `1.5x`
относительно оптимизированного serial path. Если цель не достигнута, агент
должен предоставить stage profile и объяснить bottleneck до принятия изменения.

## 6. Этап 4: специализация best-anchor alignment

Default `min_logfpr <= 0` дает максимум один query anchor и один target anchor
на strand/row. Общий CSR candidate-deduplication path для этого режима
избыточен.

Реализовать отдельный best-anchor kernel:

1. Не выделять `CandidateScratch`, dense marks или candidate vector.
2. Обрабатывать максимум два скалярных кандидата.
3. Query candidate всегда обрабатывать первым.
4. Target anchor realign-ить по текущим правилам и добавлять только при
   несовпадении с query candidate.
5. Сохранять текущий window-fit behavior и metric accumulation order.
6. Общий CSR scratch path оставить для threshold mode.

В Python переиспользовать один `AlignmentWorkspace` между orientations. В Julia
переиспользовать worker-local scratch между shifts и targets там, где остается
общий threshold path.

Проверить все пять metrics, четыре orientations, positive/negative shifts,
boundary realignment, duplicate candidates, empty rows и tie cases.

Специализацию оставлять только при ускорении prepared alignment минимум на 20%
без увеличения end-to-end allocations.

## 7. Этап 5: сократить Python Numba launch overhead

Текущий Python alignment может запускать до
`21 shifts x 4 orientations = 84` kernels на один target. После этапов 2-4
повторно измерить долю kernel-launch overhead.

Если overhead значим, объединить shifts одной orientation в один Numba kernel:

- `prange` по независимым rows;
- shifts обрабатывать внутри row;
- хранить partial values отдельно для каждого `(shift, row)`;
- редуцировать rows для каждого shift в прежнем порядке;
- не использовать этот inner parallel path одновременно с outer target pool.

Учитывать память: полный partial tensor не должен создавать многогигабайтные
worker-local allocations. При необходимости обрабатывать shifts небольшими
фиксированными блоками.

Изменение принимать только при сохранении exact discrete fields, существующих
numerical tolerances и измеримом end-to-end улучшении.

## 8. Этап 6: reuse и bounded caching

Для повторных сравнений использовать prepared profiles:

- query готовить один раз на `(model, sequences, background, min_logfpr)`;
- target profile готовить не более одного раза на ту же конфигурацию;
- cache key должен включать model fingerprint, sequence fingerprint,
  background fingerprint, normalization strategy и `min_logfpr`;
- disk/memory cache должен иметь явный size limit;
- не удерживать автоматически десятки гигабайт normalized profiles.

Null construction должен сравнивать prepared-to-prepared profiles и не
повторять scan/normalization одного model для каждой eligible pair.

## 9. Полная проверка

Python:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/mimosa
uv run pytest
```

Julia:

```bash
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_parallel.jl")'

julia --project=Mimosa.jl/test/downstream \
  Mimosa.jl/test/downstream/runtests.jl

julia --project=Mimosa.jl/test -e \
  'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'
```

После correctness suite повторить cross-language benchmark на свободной машине
с теми же входами, thread counts и runtime versions.

## 10. Definition of Done

Работа завершена, когда выполнены все условия:

1. Python и Julia проходят frozen compatibility fixtures.
2. Serial и threaded outputs имеют одинаковый порядок и exact discrete fields.
3. Common-case empirical normalization ускорена минимум в два раза.
4. Four-thread 1-vs-50 показывает подтвержденное target-level scaling.
5. End-to-end 1-vs-50 существенно быстрее baseline `25.552 s` для Python и
   `18.594 s` для Julia.
6. Scan benchmarks не регрессируют более чем на 10% без отдельного объяснения и
   одобрения.
7. Нет unbounded caches, nested parallelism или shared mutable worker buffers.
8. README, changelog и benchmark methodology обновлены только фактическими
   результатами.
9. Все изменения отформатированы, протестированы и сопровождаются before/after
   benchmark report.
