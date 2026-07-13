# План архитектурного рефакторинга Mimosa.jl

Статус: черновик для поэтапной реализации.

Дата ревизии: 2026-07-13.

## 1. Цели

- Сделать границы подсистем и публичного API очевидными без искусственного
  разбиения пакета на множество Julia-модулей.
- Устранить небезопасные и неоднозначные контракты в `src/scanning/`.
- Зафиксировать числовые и геометрические контракты до изменения ядер
  сканирования.
- Уменьшить количество файлов, не несущих самостоятельной ответственности,
  одновременно разделив самые крупные файлы по назначению.
- Убрать дублирование там, где его можно заменить multiple dispatch,
  стандартными интерфейсами Julia и общими внутренними утилитами.
- Сохранить численную совместимость, порядок результатов, правила разрешения
  ничьих, форматы хранения и существующую модель параллелизма.

## 2. Не цели

- Не переносить научную логику в CLI.
- Не возвращать удаленное прямое сравнение матриц мотивов, метрики PCC или
  Euclidean и null-стратегию `"motif"`.
- Не заменять плоские `EncodedSequenceBatch` и `RaggedArray` на
  `Vector{Vector}`.
- Не создавать подмодуль Julia для каждого каталога.
- Не менять порядок Float32-вычислений, редукций и разрешения ничьих ради
  сокращения кода.
- Не упрощать собственный планировщик до `Threads.@threads`: текущие контракты
  требуют ограничения числа задач, взвешенного разбиения, защиты от вложенного
  параллелизма и детерминированного порядка.
- Не менять форматы model/null/cache bundles без отдельного решения о версии
  формата и миграции.
- Не изменять Python-пакет, кроме отдельной задачи на межъязыковую
  совместимость.

## 3. Обязательные решения до рефакторинга

Эти решения блокируют изменения сканирующих ядер. Их нужно оформить в ADR или
в `docs/src/numerical_compatibility.md`, затем закрепить тестами.

### 3.1. Тип результатов сканирования

Сейчас модели допускают `T <: AbstractFloat`, PWM частично сохраняет `T`, а
higher-order пути выделяют `Float32`. Это создает разные контракты и приводит к
`MethodError` для некоторых сочетаний модели и выходного буфера.

Предпочтительное решение:

- сделать `Float32` каноническим типом матриц, фона и результатов
  сканирования;
- приводить входные числовые данные к `Float32` в конструкторах моделей или на
  границе I/O;
- явно отклонять выходные буферы неподдерживаемого типа;
- оставить Float64 только там, где это требует задокументированный контракт:
  межколоночные суммы метрик, GEV fitting и survival calculations.

Альтернатива: сделать все сканирование действительно параметрическим по `T`.
Ее следует выбирать только при наличии практического сценария и benchmark,
поскольку она расширяет поверхность API и число специализаций.

### 3.2. Геометрия higher-order моделей

Нужно однозначно определить число позиций для BaMM, SiteGA, Dimont и Slim:

- по полной ширине окна модели; или
- по длине мотива без дополнительного контекста.

Текущие Julia-код, Python reference и генератор frozen fixtures дают не во всех
местах одинаковый ответ, а существующие compatibility tests проверяют срезы и
не фиксируют точную длину строк и масок.

До решения нельзя менять `npositions_ho` и соответствующие смещения. После
решения необходимо:

- описать формулу и смысл координат;
- добавить точные проверки длины каждой строки, масок и граничных случаев;
- сверить координаты forward и reverse-complement результатов;
- обновить документацию и тесты, не перегенерируя frozen fixtures только ради
  прохождения регрессии.

### 3.3. Публичная поверхность низкоуровневого API

Определить, какие функции являются стабильным публичным API:

- `scan` и `scan!` остаются публичной безопасной границей;
- model-specific `npositions_*` заменить общим `npositions(model, seq_len)`;
- низкоуровневые `scan_forward!`, `scan_reverse!`, `scan_best!` и
  `scan_both!` либо сделать внутренними, либо документировать как безопасные
  публичные функции со всеми проверками;
- устаревшие публичные имена удалять только через документированный цикл
  deprecation, если они входят в downstream contract.

## 4. Целевая организация исходников

Оставить один модуль `Mimosa`. Каталоги должны группировать ответственности, а
не вводить отдельные пространства имен.

```text
src/
  Mimosa.jl                 # порядок include и единственный список export
  core/                     # ошибки и execution policies
  models/                   # типы моделей и их инварианты
  sequences/                # кодирование и плоские контейнеры
  scanning/
    interface.jl            # scan/scan!, npositions, проверки границы API
    strands.jl              # StrandPolicy и dispatch
    pwm_kernel.jl           # только PWM-ядра
    higher_order_kernel.jl  # только higher-order ядра
    batch.jl                # выделение ragged output и планирование batch
  comparison/
    metrics.jl
    normalization.jl
    anchors.jl
    alignment.jl
    prepared.jl
    api.jl
  sites/
  statistics/
  storage/                  # NPY, manifests, bundles, cache primitives
  io/                       # чтение/запись научных форматов
  cli/                      # parser/help, command runners, JSON boundary
```

Количество файлов не является самостоятельной метрикой качества. Ожидаемый
результат реорганизации: удалить почти пустые include-обертки и четыре
model-specific scan adapters, но разделить монолитные файлы по ответственности.
Итоговое количество файлов может остаться близким к текущему.

## 5. Этапы реализации

Каждый этап должен быть отдельным небольшим change set с собственными тестами.
Не совмещать массовое перемещение файлов с изменением численных алгоритмов.

### Этап 0. Зафиксировать baseline и контракты

- [ ] Записать Julia version, CPU, число runtime threads и текущий commit.
- [ ] Сохранить результаты focused tests для scanning, compatibility,
  parallelism и sites в serial и four-thread режимах.
- [ ] Снять baseline времени и allocations для PWM и higher-order public paths
  после прогрева компиляции.
- [ ] Принять решение о каноническом числовом типе согласно разделу 3.1.
- [ ] Провести аудит higher-order геометрии согласно разделу 3.2.
- [ ] Составить список экспортируемых имен и downstream usages.
- [ ] Зафиксировать известные full-suite failures отдельно от новых регрессий.

Результат этапа: документация контрактов и тесты, которые падают при их
нарушении, без перестройки каталогов.

### Этап 1. Сделать границу scanning безопасной

- [ ] Разделить безопасные публичные wrappers и приватные unchecked kernels.
- [ ] В `scan`/`scan!` для сырых векторов проверять допустимые DNA-коды,
  геометрию входа и размер destination до входа в `@inbounds`-ядро.
- [ ] Для всех принимаемых `AbstractVector` вызвать
  `Base.require_one_based_indexing` либо явно поддержать произвольные axes.
- [ ] Для уже проверенного `EncodedSequenceBatch` использовать внутренний
  unchecked путь, чтобы не выполнять O(n)-валидацию повторно.
- [ ] Проверять `Base.mightalias(forward, reverse)` в `scan_both!` и отклонять
  полное или частичное перекрытие выходных буферов.
- [ ] Заменить fallback `else # BestStrand` на dispatch по конкретным
  `ForwardStrand`, `ReverseStrand`, `BestStrand` и `BothStrands`.
- [ ] Для неизвестного `StrandPolicy` получать явный `MethodError` или
  `ArgumentError`, но никогда не интерпретировать его как `BestStrand`.
- [ ] Унифицировать поведение одиночного и batch scanning при ошибках.

Обязательные тесты:

- некорректные коды и несовместимая геометрия;
- короткая и пустая последовательность;
- destination неправильной длины и типа;
- вектор с не one-based axes либо его явное отклонение;
- неизвестная strand policy;
- полное и частичное aliasing forward/reverse buffers;
- отсутствие частично заполненного результата при исключении worker-задачи.

### Этап 2. Упростить dispatch и orchestration scanning

- [ ] Ввести общие accessors `scorematrix(model)`, `motif_length(model)`,
  `window_size(model)` и `scoretype(model)` вместо прямого доступа к полям
  через абстрактный тип.
- [ ] Ввести единый `npositions(model, seq_len)` с dispatch по семейству модели.
- [ ] Сохранить model-specific aliases только как deprecated wrappers, если это
  требуется публичным контрактом.
- [ ] Перенести выделение `RaggedArray`, расчет offsets/costs и запуск scheduler
  в общий batch layer.
- [ ] Удалить неиспользуемые аргументы, включая `strands` в общих
  higher-order helpers.
- [ ] Заменить closure-based выбор операции на прямой dispatch там, где это
  улучшает type inference и читаемость.
- [ ] Объединить четыре почти пустых файла `bamm_scan.jl`, `sitega_scan.jl`,
  `dimont_scan.jl` и `slim_scan.jl` в общий higher-order interface.
- [ ] Удалить дублирование `_scan_offsets`/`_ho_scan_offsets` и общих
  allocating/in-place wrappers.
- [ ] Не объединять fused forward/reverse/best/both inner loops только ради DRY:
  изменение допустимо лишь при сохранении порядка операций и подтвержденном
  benchmark.

Критерии этапа:

- `@inferred` для single и batch calls каждой strand policy;
- отсутствие `Any` в проверяемых hot paths;
- warmed kernels не получают новых allocations;
- serial и threaded результаты совпадают по порядку, discrete fields и
  требуемой точности значений;
- производительность public path не ухудшается без документированной причины.

### Этап 3. Довести контейнеры до ясного Julia-интерфейса

- [ ] Добавить `Base.require_one_based_indexing` в конструкторы, которые
  предполагают one-based vectors.
- [ ] Реализовать для `EncodedSequenceBatch` и `RaggedArray` согласованный
  минимальный интерфейс: `getindex`, `firstindex`, `lastindex`, `iterate`,
  `length` и корректные iterator traits.
- [ ] Не делать типы подтипами `AbstractVector`, пока не реализован полный
  ожидаемый контракт коллекции.
- [ ] Вынести повторяющееся flattening строк в одну `_flatten_rows` на основе
  предварительного расчета offsets и `copyto!`.
- [ ] Убрать `Vector{Vector}` staging из генерации случайных batch,
  batch reverse complement и `from_padded`, сохранив порядок RNG-вызовов.
- [ ] Удалить повторную валидацию offsets между `_unsafe_encoded_batch` и
  внутренним конструктором.
- [ ] Проверить сохранение пустых строк, порядка строк и точных offsets.

Отдельно пересмотреть модельный интерфейс:

- [ ] Не использовать одновременно `Base.length(model)` как ширину мотива и
  `Base.size(model)` как форму внутренней матрицы: это нарушает ожидание
  `length == prod(size)`.
- [ ] Предпочесть явные `motif_length`, `window_size` и `scorematrix`.
- [ ] Если `length`/`size` уже публичны, сначала добавить deprecation и обновить
  downstream contract.

### Этап 4. Выровнять архитектурные границы

- [ ] Объединить концептуально циклические `comparison/` и `profiles/` в одну
  подсистему profile comparison с направлением зависимостей:
  metrics -> normalization/anchors -> alignment -> prepared -> public API.
- [ ] Оставить единственный export block в `src/Mimosa.jl`.
- [ ] Убрать exports из include manifests и не экспортировать внутренние имена
  с `_`, включая `_fit_transform_empirical`.
- [ ] Добавить тест или явный snapshot стабильной публичной поверхности.
- [ ] Сгруппировать bundle primitives, model/null storage, cache manifests и
  fingerprints в `storage/`, сохранив bounded validation, checksum verification
  и atomic writes.
- [ ] Разделить `cli.jl` на parser/help, command runners и JSON serialization,
  не перенося в CLI научную логику.
- [ ] Удалить дублирование JSON encoding между CLI и portable serialization.
- [ ] Удалить include-файлы, которые не задают API, порядок зависимостей или
  самостоятельную ответственность.
- [ ] Разделить крупные файлы `profiles/alignment.jl`, `sites/sites.jl`,
  `io/bundle_storage.jl` и `cli.jl` по ответственности, а не по числу строк.

Перемещения выполнять отдельно от функциональных изменений, чтобы review diff
оставался проверяемым.

### Этап 5. Убрать локальную избыточность

Все замены сначала подтвердить unit/compatibility tests и, для hot paths,
benchmark.

- [ ] Рассмотреть замену `_argmax_first` на `findmax`, сохранив выбор первого
  элемента при равенстве.
- [ ] Рассмотреть замену `_lower_bound_desc` на `searchsortedfirst(...;
  rev=true)` с прежней обработкой краев.
- [ ] Использовать `diff(offsets)` для простых расчетов длины строк, если это не
  увеличивает allocations на hot path.
- [ ] Объединить повторяющиеся проверки score bounds и построение ragged output.
- [ ] Заменить algorithm-critical mutable global tables на immutable tuples или
  функции.
- [ ] Представить CLI option specs immutable tuples/NamedTuples вместо
  глобальных изменяемых `Dict`/`Set`, где это не ухудшает parser.
- [ ] Не заменять численные циклы на `dot`, BLAS, `sum` с иным деревом редукции,
  `@simd` или `@fastmath` без доказательства совместимости.

Собственные типы, которые следует сохранить:

- `EncodedSequenceBatch` и `RaggedArray` для плоского ragged layout;
- `StrandPolicy` и конкретные policy types для dispatch вместо строк/symbols;
- `StrandPair`, `PreparedProfile`, `AnchorCSR` и execution policy types;
- bounded dynamic scheduler и weighted scheduler.

### Этап 6. Обновить документацию

- [ ] Удалить из README и architecture docs утверждения о прямом motif matrix
  comparison и удаленных CLI/API.
- [ ] Исправить ссылку из `src/Mimosa.jl` на отсутствующие `REFACTORING.md` и
  `PLAN.md`.
- [ ] Описать фактический include order, profile-only workflow и новую структуру
  scanning.
- [ ] Обновить API docs, CLI help, data layout, numerical compatibility,
  reproducibility и storage docs только в затронутых частях.
- [ ] Обновить `[Unreleased]` в `CHANGELOG.md` для публичных изменений.
- [ ] Обновить downstream contract при deprecation или изменении exports.
- [ ] Проверить Documenter export checking и добавить отсутствующие docstrings
  для оставшихся публичных имен.

## 6. Проверки после каждого этапа

Минимальный набор:

```bash
julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_validation.jl")'

julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_parallel.jl")'

JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/test -e \
  'using Mimosa, Test; include("Mimosa.jl/test/unit/test_parallel.jl")'
```

Дополнительно для scanning выполнять focused tests BaMM, SiteGA, Dimont, Slim,
PWM и compatibility fixtures. Имена фактических test files нужно сверять перед
запуском, а не копировать из устаревшей документации.

Перед завершением этапа:

- [ ] BlueStyle formatting check для затронутых source/test files.
- [ ] Focused serial tests с `SerialExecution()`.
- [ ] Focused four-thread tests с явным `ThreadedExecution(...)`.
- [ ] Compatibility tests без ослабления tolerances и без перегенерации frozen
  fixtures.
- [ ] Aqua/JET или более узкая inference-проверка для затронутого API.
- [ ] Documentation build при изменении exports или docs.
- [ ] Benchmark до/после на одинаковом input, seed, Julia version, CPU и числе
  runtime threads; сообщать median time и allocations.
- [ ] Более широкий `Pkg.test()` по возможности, с отдельным учетом известных
  baseline failures и новых регрессий.

В ограниченной среде использовать writable depot prefix, не скрывая
установленный depot:

```bash
JULIA_DEPOT_PATH=/tmp/mimosa-julia-depot:$HOME/.julia julia ...
```

## 7. Порядок приоритетов

1. Научные контракты higher-order геометрии и Float32.
2. Безопасность публичной границы scanning.
3. Явный dispatch strand policies и единый scanning interface.
4. Тесты точной геометрии, inference, aliasing и ошибок.
5. Контейнеры и устранение промежуточных `Vector{Vector}`.
6. Консолидация exports и архитектурных границ.
7. Перемещение файлов, упрощение CLI/storage и локальные Base-замены.
8. Полная синхронизация документации.

## 8. Критерии завершения

Рефакторинг считается завершенным, когда одновременно выполнены условия:

- публичные `scan`/`scan!` не могут передать непроверенные коды, геометрию или
  aliasing buffers в `@inbounds`-ядра;
- числовой тип результатов и формула `npositions` едины и документированы;
- неизвестные policy types не получают неявную семантику;
- single и batch scanning используют один согласованный interface;
- hot kernels type-stable и не получили лишних allocations;
- serial/threaded execution сохраняют порядок и совместимые результаты;
- frozen compatibility assertions и правила координат соблюдены;
- public exports определены в одном месте и покрыты документацией;
- структура каталогов отражает направления зависимостей;
- нет почти пустых model-specific scan adapters и бессодержательных include
  layers;
- сокращение дублирования не изменило порядок численных операций;
- focused tests, formatter, relevant Aqua/JET checks и docs build проходят;
- результаты benchmark и известные residual full-suite failures зафиксированы в
  changelog или отчете изменения.

## 9. Стратегия поставки

Предпочтительны небольшие последовательные PR/коммиты:

1. contract tests и документация без изменения поведения;
2. safety wrappers и alias checks;
3. dispatch/interface consolidation;
4. container/data-movement improvements;
5. directory/export/storage/CLI reorganization;
6. локальные упрощения и финальная документация.

Каждый change set должен оставлять пакет работоспособным. Не объединять в одном
diff массовые перемещения, изменение public API и переписывание численного ядра:
иначе невозможно надежно отличить архитектурный рефакторинг от научной
регрессии.
