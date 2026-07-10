Ты — ведущий Julia-разработчик и архитектор scientific software. Твоя задача — спроектировать и выполнить качественный перенос проекта MIMOSA с Python на Julia.

Исходный проект: mimosa

Новый проект должен называться `Mimosa.jl`.

Это не задача по построчному переводу Python-кода. Необходимо переосмыслить архитектуру с учётом особенностей Julia:

* multiple dispatch;
* параметрические типы;
* type stability;
* специализация методов;
* эффективные обычные циклы;
* views и контроль аллокаций;
* column-major layout;
* встроенная многопоточность;
* Julia package environments;
* package extensions и weak dependencies;
* artifacts для внешних бинарных зависимостей;
* precompilation;
* особенности latency и распространения CLI-приложений.

Главная цель — создать самостоятельный, читаемый, производительный и расширяемый Julia-пакет, а не Python-программу, синтаксически переписанную на Julia.

# 1. Главные требования

Новая реализация должна:

1. Сохранять научный смысл и основные пользовательские возможности MIMOSA.
2. Иметь чётко выделенное библиотечное ядро, независимое от CLI.
3. Использовать идиоматичную архитектуру Julia.
4. Не использовать `Any`, `Dict{String,Any}` и строковые type keys в горячих вычислительных путях.
5. Не копировать Python-паттерны, возникшие из-за ограничений Python, NumPy или Numba.
6. Не использовать избыточную объектную иерархию в стиле Java/Python OOP.
7. Не превращать каждый Python-класс в Julia `mutable struct`.
8. Не векторизовать код искусственно там, где обычные Julia-циклы проще и быстрее.
9. Не оптимизировать преждевременно без benchmark и profiling.
10. Сначала обеспечить корректность и понятную архитектуру, затем производительность.
11. Поддерживать воспроизводимые результаты и явно документировать случаи, где битовая совместимость с Python невозможна.
12. Иметь полноценные unit, property, integration, compatibility и benchmark tests.
13. Обеспечивать безопасную обработку пользовательских файлов.
14. Не использовать Python `pickle` или `joblib` как основной формат хранения.
15. Иметь версионированный, документированный и переносимый формат моделей и null distributions.
16. Быть пригодной для дальнейшего использования из проекта `MotifHORDE.jl`.

# 2. Обязательный предварительный анализ

Перед написанием кода исследуй текущий Python-репозиторий.

Не начинай с создания Julia-файлов, пока не составишь карту существующей системы.

Определи:

* публичный Python API;
* CLI-команды и их параметры;
* форматы входных и выходных файлов;
* поддерживаемые типы motif models;
* внутренние model handlers;
* алгоритмы сканирования;
* алгоритмы сравнения;
* profile-based и motif-based режимы;
* способы обработки forward/reverse strands;
* semantics offsets и orientations;
* алгоритмы извлечения sites;
* реконструкцию PFM;
* способы нормализации score profiles;
* null distributions;
* GEV fitting;
* p-value и FDR calculations;
* кэширование;
* распараллеливание;
* использование NumPy, SciPy, pandas, Numba и joblib;
* существующие тесты и fixtures;
* сериализованные форматы;
* точки расширения пользовательскими моделями;
* API, которым потенциально пользуется MotifHORDE.

Составь документ `docs/python_reference_architecture.md`, в котором отрази:

* карту Python-модулей;
* их ответственность;
* публичные контракты;
* вычислительные зависимости;
* участки, которые нельзя переносить буквально;
* предполагаемое соответствие компонентам Julia;
* риски численной несовместимости.

Не считай существующую архитектуру Python автоматически правильной. Разделяй:

* предметные ограничения;
* исторические решения;
* обходные пути для Python;
* ограничения Numba;
* случайные особенности реализации.

# 3. Архитектурные принципы Julia

## 3.1. Библиотека прежде CLI

Основной пакет должен предоставлять полноценный Julia API.

CLI должен быть тонким адаптером:

```text
CLI arguments
    ↓
typed configuration
    ↓
public Mimosa API
    ↓
domain and numerical kernels
    ↓
typed result
    ↓
JSON/text serialization
```

В CLI не должно находиться:

* алгоритмов сканирования;
* сравнения моделей;
* вычисления статистик;
* обработки массивов;
* логики model dispatch;
* парсинга motif formats, кроме вызова публичного API.

## 3.2. Предметные типы

Не используй единую структуру следующего типа:

```julia
struct GenericModel
    type_key::String
    representation::Any
    metadata::Dict{String,Any}
end
```

Вместо этого спроектируй предметную систему типов.

Базовое направление:

```julia
abstract type AbstractMotifModel end

abstract type AbstractMatrixMotif <: AbstractMotifModel end
abstract type AbstractHigherOrderMotif <: AbstractMotifModel end
abstract type AbstractScanner end
abstract type AbstractComparator end
abstract type AbstractMetric end
abstract type AbstractSiteSelector end
```

Конкретные модели должны иметь конкретные и параметрические поля:

```julia
struct PWM{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractMatrixMotif
    name::String
    weights::M
    background::NTuple{4,T}
end
```

Это только ориентир, а не обязательная финальная форма.

Реши отдельно, должны ли:

* PFM и PWM быть разными типами;
* PFM быть представлением, а не полноценной model type;
* BaMM хранить данные в трёхмерном массиве или специализированной структуре;
* metadata быть параметрическим типом;
* model provenance быть отдельной структурой;
* внешние модели содержать исходные параметры discovery tool.

Не добавляй поля «на всякий случай».

## 3.3. Multiple dispatch вместо registry callbacks

Типичное поведение должно выражаться методами:

```julia
scan(model::PWM, sequences, config)
scan(model::BaMM, sequences, config)

scorebounds(model::PWM)
scorebounds(model::BaMM)

sites(model::PWM, sequences, selector)
sites(model::SiteGA, sequences, selector)

reconstruct_pfm(model::AbstractMotifModel, sequences, config)
```

Не создавай центральный словарь:

```julia
Dict("pwm" => scan_pwm, "bamm" => scan_bamm)
```

Строковые идентификаторы допустимы только:

* при чтении файлов;
* при CLI parsing;
* в сериализованном metadata;
* при отображении результата пользователю.

Сразу после пересечения I/O-границы они должны преобразовываться в конкретные Julia-типы.

## 3.4. Composition вместо глубокой иерархии

Не создавай глубокую иерархию абстрактных типов только ради классификации.

Разделяй независимые аспекты композиционно:

* motif representation;
* scanner;
* strand policy;
* normalization;
* site selection;
* comparison metric;
* null model;
* parallel execution policy.

Например:

```julia
struct ScanConfig{S,N,T}
    strands::S
    normalization::N
    score_type::Type{T}
end
```

или другая type-stable структура.

Не кодируй все параметры через `Symbol`, если конкретный маленький тип улучшает ясность и dispatch.

Например, предпочтительнее:

```julia
abstract type StrandPolicy end
struct ForwardOnly <: StrandPolicy end
struct BestStrand <: StrandPolicy end
struct BothStrands <: StrandPolicy end
```

чем многократные проверки:

```julia
if strand == :best
```

Но не создавай тип ради каждого булевого параметра без практической пользы. Поддерживай разумный баланс.

# 4. Предлагаемая организация проекта

Используй стандартную структуру Julia-пакета.

```text
Mimosa.jl/
├── Project.toml
├── Manifest.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   ├── Mimosa.jl
│   ├── api.jl
│   ├── errors.jl
│   ├── models/
│   │   ├── models.jl
│   │   ├── pwm.jl
│   │   ├── pfm.jl
│   │   ├── bamm.jl
│   │   ├── sitega.jl
│   │   ├── dimont.jl
│   │   └── slim.jl
│   ├── sequences/
│   │   ├── sequences.jl
│   │   ├── encoding.jl
│   │   ├── batches.jl
│   │   └── reverse_complement.jl
│   ├── scanning/
│   │   ├── scanning.jl
│   │   ├── interface.jl
│   │   ├── pwm_scan.jl
│   │   ├── higher_order_scan.jl
│   │   └── strands.jl
│   ├── profiles/
│   │   ├── profiles.jl
│   │   ├── ragged.jl
│   │   ├── normalization.jl
│   │   ├── anchors.jl
│   │   └── windows.jl
│   ├── comparison/
│   │   ├── comparison.jl
│   │   ├── metrics.jl
│   │   ├── alignment.jl
│   │   ├── matrix_comparison.jl
│   │   ├── profile_comparison.jl
│   │   └── results.jl
│   ├── sites/
│   │   ├── sites.jl
│   │   ├── selectors.jl
│   │   └── reconstruction.jl
│   ├── statistics/
│   │   ├── statistics.jl
│   │   ├── null_distributions.jl
│   │   ├── gev.jl
│   │   ├── pvalues.jl
│   │   └── multiple_testing.jl
│   ├── io/
│   │   ├── io.jl
│   │   ├── fasta.jl
│   │   ├── meme.jl
│   │   ├── pfm.jl
│   │   ├── bamm.jl
│   │   ├── sitega.jl
│   │   ├── xml_models.jl
│   │   ├── stored_model.jl
│   │   └── null_format.jl
│   ├── parallel/
│   │   ├── parallel.jl
│   │   ├── policies.jl
│   │   └── scheduling.jl
│   ├── cache/
│   │   ├── cache.jl
│   │   └── keys.jl
│   ├── cli/
│   │   ├── cli.jl
│   │   ├── arguments.jl
│   │   ├── commands.jl
│   │   └── output.jl
│   └── precompile.jl
├── ext/
├── test/
│   ├── runtests.jl
│   ├── unit/
│   ├── properties/
│   ├── compatibility/
│   ├── integration/
│   ├── cli/
│   └── fixtures/
├── benchmark/
│   ├── Project.toml
│   ├── benchmarks.jl
│   └── baselines/
├── docs/
│   ├── Project.toml
│   ├── make.jl
│   └── src/
├── scripts/
│   ├── generate_python_reference.jl
│   ├── convert_legacy_model.py
│   └── convert_legacy_null.py
└── deps/
```

Это ориентир, а не требование создать множество файлов немедленно.

Не создавай файл, если в нём будет несколько строк без ясной архитектурной причины. Начни компактнее и разделяй модули по мере роста.

Избегай одного гигантского `Mimosa.jl`, содержащего всю реализацию.

Верхний модуль должен преимущественно:

* подключать подмодули;
* экспортировать тщательно выбранный API;
* не содержать реализацию алгоритмов.

# 5. Публичный API

Спроектируй небольшой и стабильный публичный API.

Примерное направление:

```julia
readmodel(path; format=:auto)
writemodel(path, model; format=:auto)

readsequences(path)
scan(model, sequences; kwargs...)
scan!(destination, model, sequences; kwargs...)

scorebounds(model)

selectsites(model, sequences, selector)
reconstruct_pfm(model, sequences; kwargs...)

compare(query, target; kwargs...)
compare(query, target, sequences; kwargs...)

build_null(models; kwargs...)
pvalue(null_distribution, score)

save_null(path, distribution)
load_null(path)
```

Имена должны соответствовать Julia conventions:

* функции в нижнем регистре;
* без избыточных `get_`;
* `!` только для функций, изменяющих аргументы;
* возвращаемые структуры документированы;
* positional arguments — для основных сущностей;
* keyword arguments — для конфигурации.

Не экспортируй внутренние helper functions.

Раздели:

* public API;
* extension API для добавления моделей;
* internal API.

Опиши extension API в `docs/src/extending_models.md`.

# 6. Представление последовательностей

Проанализируй, что эффективнее для конкретных алгоритмов:

* `BioSequences.LongDNA`;
* собственное кодирование `UInt8`;
* packed representation;
* ленивое чтение FASTA;
* плоский буфер с offsets;
* `Vector{Vector{UInt8}}`;
* batch representation.

Не принимай решение только из соображений эстетики.

Для сканирования больших наборов последовательностей предпочтительны:

* компактное кодирование;
* последовательный доступ к памяти;
* минимум преобразований;
* отсутствие создания строк во внутренних циклах;
* явное поведение для `N` и других ambiguous bases.

Определи и документируй:

* кодирование A/C/G/T;
* ambiguous bases;
* lowercase;
* пустые последовательности;
* последовательности короче motif width;
* reverse-complement semantics;
* координатную систему;
* включительность границ;
* zero-based или one-based координаты во внешнем JSON.

Внутри Julia используй естественную one-based индексацию.

Не сохраняй zero-based внутреннюю индексацию только ради соответствия Python. Преобразование должно происходить на I/O-границе.

# 7. Массивы и layout

Учитывай column-major layout Julia.

Не копируй форму NumPy-массивов автоматически. Для каждой структуры реши, какой dimension должен изменяться во внутреннем цикле.

Например, для PWM сравни варианты:

```julia
weights[base, position]
```

и:

```julia
weights[position, base]
```

Выбери layout на основании:

* характера сканирующего цикла;
* последовательности memory access;
* удобства reverse complement;
* BLAS здесь обычно не является главным фактором;
* необходимости interoperability.

Зафиксируй выбранные conventions в `docs/src/data_layout.md`.

Используй:

* `@views` или явный `view` там, где это действительно исключает копию;
* `eachindex`;
* `axes`;
* `similar`;
* preallocation;
* in-place APIs для горячих операций.

Не злоупотребляй `@view` для очень маленьких участков, если объект view создаёт overhead.

Не применяй `@inbounds`, `@simd`, LoopVectorization или generated functions, пока:

1. корректность не покрыта тестами;
2. benchmark не показывает необходимость;
3. profiling не подтверждает bottleneck;
4. проверена безопасность индексов.

Каждое использование `@inbounds` должно находиться в маленьком локальном kernel с явно доказанными границами.

# 8. Ragged profiles

Не копируй Python representation ragged arrays автоматически.

Рассмотри структуру:

```julia
struct RaggedArray{T,V<:AbstractVector{T},I<:AbstractVector{Int}}
    data::V
    offsets::I
end
```

или эквивалентную.

Она должна обеспечивать:

* компактное хранение;
* O(1) доступ к отдельной строке;
* `view`, а не копирование;
* типостабильную итерацию;
* проверку offsets в constructor;
* отсутствие `Vector{Any}`;
* удобную параллельную запись при заранее известных размерах.

Не реализуй полный аналог NumPy ndarray. Реализуй только операции, реально необходимые MIMOSA.

# 9. Метрики и comparison algorithms

Метрики должны быть отдельными типами или ясно организованными методами, а не строковыми ветвлениями внутри каждого горячего цикла.

Пример:

```julia
abstract type AbstractColumnMetric end

struct PearsonCorrelation <: AbstractColumnMetric end
struct EuclideanDistance <: AbstractColumnMetric end
struct CosineSimilarity <: AbstractColumnMetric end
```

Метод:

```julia
metric(metric::PearsonCorrelation, x, y)
```

или более удачная форма по Julia conventions.

Нужно определить:

* mathematical direction: similarity или distance;
* диапазон значений;
* поведение при нулевой дисперсии;
* поведение при нулевой норме;
* NaN policy;
* aggregation across columns;
* weighting;
* minimum overlap;
* orientation;
* offset convention;
* tie-breaking.

Tie-breaking должен быть детерминированным и документированным.

Например, при равных score определить приоритет:

1. больший overlap;
2. меньший абсолютный offset;
3. forward orientation;
4. лексикографический или фиксированный порядок.

Не полагайся на случайный порядок iteration или scheduling threads.

# 10. Scanning API

Предусмотри выделяющий и невыделяющий API:

```julia
scores = scan(model, sequence)
scan!(scores, model, sequence)
```

Для batch scanning:

```julia
scan(model, sequences; strands=BestStrand())
```

Определи тип результата отдельно.

Не возвращай неструктурированный tuple, содержащий множество массивов с неочевидным смыслом.

Например:

```julia
struct ScanResult{T,S,O}
    scores::S
    orientations::O
    motif_width::Int
    metadata::ScanMetadata{T}
end
```

Но не перегружай горячий путь metadata, если он не нужен для каждого вызова.

Раздели:

* минимальный kernel result;
* пользовательский result;
* CLI serialization result.

# 11. Parallelism

Не копируй `joblib` API.

Спроектируй Julia-native parallel execution.

Разделяй:

* последовательное выполнение;
* thread-based execution;
* process-based execution;
* возможное распределённое выполнение в будущем.

Не запускай threads во всех функциях автоматически.

Основные kernels должны быть последовательными и composable.

Параллелизм должен находиться на верхнем независимом уровне:

* последовательности;
* target models;
* model pairs;
* null comparisons.

Создай execution policy:

```julia
abstract type ExecutionPolicy end
struct SerialExecution <: ExecutionPolicy end
struct ThreadedExecution <: ExecutionPolicy
    ntasks::Int
end
```

или эквивалентный интерфейс.

Требования:

* детерминированный порядок результатов;
* отсутствие `push!` в общий массив из нескольких threads;
* заранее выделенные output buffers;
* thread-local scratch buffers;
* контроль вложенного parallelism;
* `BLAS.set_num_threads(1)` там, где внешний parallelism предпочтительнее;
* независимые RNG streams;
* отсутствие глобального mutable state;
* отсутствие зависимости результата от числа threads.

Не обещай thread safety без тестов.

Добавь тесты, сравнивающие serial и threaded execution.

# 12. Randomness и воспроизводимость

Все случайные операции должны принимать `AbstractRNG`.

Не используй глобальный RNG внутри библиотечных функций.

Предпочтительный API:

```julia
build_null(rng, models; kwargs...)
```

или keyword `rng`.

Для параллельных задач создавай воспроизводимые независимые streams на основании:

* базового seed;
* стабильного task index;
* стабильного идентификатора pair.

Не используй `hash` без анализа стабильности между Julia sessions и versions, если результат должен воспроизводиться между запусками.

При необходимости реализуй явное стабильное seed derivation.

Документируй:

* reproducibility within one Julia version;
* reproducibility across thread counts;
* reproducibility across Julia versions;
* отличие от NumPy RNG.

# 13. GEV fitting и статистическая совместимость

Это один из наиболее рискованных компонентов.

Нельзя просто заменить:

```python
scipy.stats.genextreme.fit(...)
```

на внешне похожую Julia-функцию и считать результат совместимым.

Исследуй:

* параметризацию shape;
* знак shape parameter;
* location и scale;
* likelihood;
* constraints;
* initial estimates;
* optimizer;
* tolerance;
* convergence handling;
* degenerate samples;
* identical values;
* small samples;
* NaN и Inf;
* upper-tail calculation;
* numerical stability survival function.

Создай отдельный compatibility corpus, содержащий:

* исходные null samples;
* параметры SciPy;
* SciPy survival probabilities;
* ожидаемые edge-case failures.

Реализация Julia должна иметь чёткий режим:

```julia
fit_gev(samples; method=NativeGEVFit())
```

Если полная совместимость со SciPy невозможна, не скрывай это.

Предоставь:

* native statistically sound implementation;
* документированные tolerances;
* migration note;
* при необходимости временный compatibility path вне основного runtime.

Не добавляй обязательную зависимость от PythonCall в core package.

Python compatibility может быть отдельным extension или migration script.

# 14. Форматы хранения

Не используй Julia `Serialization` как долговременный пользовательский формат.

Он допустим только для:

* временного локального кэша;
* данных, привязанных к точной Julia/environment version;
* явно непереносимых внутренних артефактов.

Основной формат модели должен быть:

* версионированным;
* документированным;
* проверяемым;
* по возможности language-neutral;
* безопасным при чтении;
* пригодным для будущего MotifHORDE.jl.

Рассмотри:

* JSON3 + binary arrays;
* MessagePack;
* HDF5;
* Arrow для табличных частей;
* TOML metadata + отдельные binary blobs;
* собственный простой контейнер с magic header.

Формат должен содержать:

```text
format version
model type
model name
representation
numeric dtype
shape
background
model-specific parameters
provenance
tool versions
creation timestamp при необходимости
coordinate conventions
optional calibration data
checksum при необходимости
```

Не включай в формат runtime Julia type names как единственный способ определения схемы.

Разработай миграционную стратегию:

* converter Python pickle/joblib → новый формат;
* отдельный Python script в `scripts/`;
* запрет загрузки недоверенных pickle без явного подтверждения;
* fixtures для старых моделей;
* round-trip tests.

Null distribution должна иметь отдельную схему.

# 15. Кэширование

Кэш не должен быть скрытым глобальным словарём.

Определи:

* что именно кэшируется;
* ключ;
* версию алгоритма;
* версию модели;
* параметры comparison;
* sequences identity;
* numerical dtype;
* invalidation rules.

Ключ должен быть основан на стабильной сериализации параметров, а не на `objectid` или session-dependent hash.

Кэш должен:

* быть отключаемым;
* иметь явную директорию;
* не влиять на корректность;
* переживать частично записанные файлы;
* использовать atomic write;
* уметь очищаться через CLI;
* не использовать небезопасную десериализацию.

# 16. Обработка ошибок

Создай небольшую и осмысленную иерархию исключений:

```julia
abstract type MimosaError <: Exception end

struct ModelFormatError <: MimosaError
    path::String
    message::String
end
```

Не создавай отдельный exception type для каждого частного случая.

Ошибки должны разделять:

* invalid user input;
* unsupported format;
* malformed model;
* incompatible dimensions;
* statistical fit failure;
* external I/O failure;
* internal invariant violation.

CLI должен:

* писать пользовательские ошибки в stderr;
* возвращать ненулевой exit code;
* не показывать полный stack trace по умолчанию;
* иметь verbose/debug режим;
* выдавать machine-readable JSON только в stdout;
* не смешивать progress bars с JSON.

# 17. CLI

Сохрани основные пользовательские сценарии Python MIMOSA, но не копируй внутреннюю структуру CLI.

Предпочтительные команды:

```text
mimosa profile ...
mimosa motif ...
mimosa build-null ...
mimosa cache clear
mimosa convert-model ...
mimosa convert-null ...
mimosa inspect-model ...
```

Проверь фактический существующий CLI и сохрани совместимость там, где она разумна.

CLI должен быть тонким слоем над публичным API.

Требования:

* `--help` для каждой команды;
* строгая валидация;
* понятные defaults;
* JSON output schema;
* стабильные exit codes;
* progress в stderr;
* `--threads`;
* `--seed`;
* `--quiet`;
* `--verbose`;
* `--output`;
* отсутствие интерактивных prompts в batch mode.

Рассмотри встроенные package apps Julia и `@main`.

Выбери CLI parser после сравнения:

* ArgParse.jl;
* Comonicon.jl;
* собственный небольшой parser.

Не добавляй тяжёлую dependency без необходимости.

# 18. Package extensions и optional dependencies

Core package не должен зависеть от всего возможного ecosystem.

Используй weak dependencies и package extensions, если это оправдано.

Потенциальные optional integrations:

* DataFrames.jl;
* PythonCall.jl;
* Makie.jl;
* HDF5.jl;
* Arrow.jl;
* CUDA.jl;
* ProgressLogging.jl.

Например, базовый `compare` не должен требовать DataFrames.

Результаты должны быть обычными структурами Julia.

Интеграция:

```julia
DataFrame(results)
```

может быть реализована extension-методом.

Не добавляй GPU support только ради наличия feature. Сначала оцени применимость алгоритмов и реальные размеры workload.

# 19. Производительность

Производительность оценивай после создания корректной reference implementation.

Используй:

* BenchmarkTools.jl;
* Profile;
* ProfileView или PProf при необходимости;
* `@code_warntype`;
* `@allocated`;
* JET.jl;
* Aqua.jl;
* SnoopCompile при работе над latency.

Критические требования:

* отсутствие type instability в горячих kernels;
* отсутствие `Vector{Any}`;
* отсутствие абстрактных полей;
* отсутствие случайных аллокаций в каждом inner-loop iteration;
* preallocation для повторяющихся сравнений;
* function barriers вокруг heterogeneous collections;
* grouping models by concrete type при batch execution;
* dispatch вне самого глубокого цикла;
* отсутствие строк и `Symbol` comparisons во внутренних циклах.

При наличии:

```julia
models::Vector{AbstractMotifModel}
```

не делай dispatch на каждую позицию последовательности.

Допустимые стратегии:

* dispatch один раз на модель;
* группировка по concrete model type;
* function barrier;
* tuple/union для небольшого закрытого набора;
* type erasure wrapper только после benchmark и при ясной необходимости.

Сравни:

1. cold CLI startup;
2. warm library runtime;
3. first-call compilation;
4. repeated-call performance;
5. memory allocation;
6. peak RSS;
7. serial performance;
8. threaded scaling;
9. package precompile time;
10. standalone app size.

Не публикуй benchmark без:

* версии Julia;
* CPU;
* числа threads;
* размера данных;
* warm-up policy;
* версии Python reference;
* версии зависимостей.

# 20. Precompilation и latency

Добавь precompile workload только после стабилизации API.

Покрой representative paths:

* чтение PWM;
* чтение FASTA;
* PWM scanning;
* reverse complement;
* matrix comparison;
* profile comparison;
* null p-value;
* JSON serialization;
* CLI argument path.

Не выполняй тяжёлые реальные вычисления при `using Mimosa`.

Модуль не должен:

* читать пользовательские файлы при import;
* создавать cache directories при import;
* запускать threads при import;
* изменять глобальные настройки BLAS при import;
* печатать что-либо при import.

# 21. Тестовая стратегия

## 21.1. Unit tests

Покрыть:

* constructors и invariants;
* sequence encoding;
* reverse complement;
* parsers;
* model-specific scanning;
* score bounds;
* metrics;
* alignment;
* offset handling;
* orientation handling;
* site extraction;
* PFM reconstruction;
* null serialization;
* p-values;
* cache keys.

## 21.2. Property tests

Проверить:

* `reverse_complement(reverse_complement(x)) == x`;
* детерминированность;
* serial/threaded equivalence;
* round-trip model serialization;
* identical motif comparison;
* orientation invariance там, где она ожидается;
* offset symmetry там, где она математически ожидается;
* score bounds;
* отсутствие изменения input arrays функциями без `!`;
* equivalence allocating и in-place APIs;
* no out-of-bounds on short sequences;
* stable tie-breaking.

## 21.3. Python compatibility tests

Python-реализация должна выступать oracle, но не должна запускаться в обычных unit tests Julia-пакета.

Создай отдельный набор fixtures:

```text
input model
input FASTA
configuration
Python output
expected intermediate values
Julia output tolerance
```

Проверяй отдельно:

* parser equivalence;
* scanning tracks;
* strand selection;
* normalization;
* anchor sites;
* shifts;
* orientations;
* reconstructed PFMs;
* comparison scores;
* GEV parameters;
* p-values;
* CLI JSON.

Не ограничивай compatibility test только финальным score. Иначе источник расхождения будет трудно определить.

## 21.4. Integration tests

Покрыть реальные команды:

```text
mimosa motif
mimosa profile
mimosa build-null
mimosa cache clear
```

Проверить:

* stdout;
* stderr;
* exit code;
* output files;
* malformed input;
* empty input;
* unsupported model;
* interrupted or partial cache file.

## 21.5. Quality tooling

Добавь:

* Aqua.jl;
* JET.jl;
* JuliaFormatter.jl;
* Documenter.jl;
* coverage;
* doctests;
* CompatHelper;
* TagBot;
* Dependabot только если это не дублирует CompatHelper.

Не игнорируй warnings ради зелёного CI.

# 22. Документация

Создай:

* README с кратким quick start;
* installation;
* CLI examples;
* Julia API examples;
* поддерживаемые модели;
* формат данных;
* performance notes;
* migration from Python;
* extension guide;
* architecture;
* numerical compatibility;
* reproducibility;
* serialization format;
* security considerations.

Каждая экспортируемая сущность должна иметь docstring.

Документация должна явно объяснять:

* one-based internal indexing;
* external coordinate representation;
* orientation conventions;
* offset conventions;
* score direction;
* NaN policy;
* RNG behavior;
* GEV differences from SciPy;
* limitations.

# 23. Совместимость с MotifHORDE.jl

Mimosa.jl должен проектироваться как нижний независимый слой для будущего MotifHORDE.jl.

Не добавляй MotifHORDE-specific orchestration в Mimosa.jl.

Mimosa.jl должен владеть:

* model types;
* model readers/writers;
* scanning;
* score calibration;
* site extraction;
* PFM reconstruction;
* motif comparison;
* profile comparison;
* null distributions;
* statistical evaluation непосредственно comparison result.

MotifHORDE.jl должен в будущем владеть:

* запуском discovery tools;
* parameter grids;
* odd/even validation;
* model selection;
* full-data rerun;
* pipeline orchestration;
* output directory layout.

Подготовь стабильный API, которым MotifHORDE.jl сможет пользоваться без доступа к внутренним модулям.

Добавь contract tests для предполагаемого downstream API.

# 24. Этапы реализации

Работай вертикальными этапами.

## Этап 0. Аудит

Результаты:

* Python architecture report;
* feature matrix;
* format inventory;
* public API inventory;
* numerical risk register;
* migration plan.

Не писать production Julia kernels до завершения этого этапа.

## Этап 1. Минимальный пакет

Реализовать:

* package skeleton;
* PWM/PFM types;
* sequence encoding;
* FASTA reading;
* matrix metric primitives;
* motif alignment;
* reverse complement;
* basic result types;
* tests;
* documentation.

Цель — поддержать минимальный сценарий PWM vs PWM.

## Этап 2. PWM scanning

Реализовать:

* allocating API;
* in-place API;
* strand policies;
* batch sequences;
* short-sequence handling;
* serial execution;
* benchmark.

## Этап 3. Profile comparison

Реализовать:

* ragged profiles;
* normalization;
* anchors;
* windows;
* shifts;
* realignment;
* profile metrics;
* deterministic tie-breaking.

## Этап 4. Site extraction и PFM reconstruction

Реализовать:

* selectors;
* threshold-based selection;
* top-fraction selection;
* orientation-aware site extraction;
* reconstruction;
* minimum site constraints.

## Этап 5. Дополнительные модели

Добавлять по одной model family:

* BaMM;
* SiteGA;
* Dimont;
* Slim или фактические модели исходного проекта.

Для каждой модели обязательны:

* concrete type;
* parser;
* writer, если применимо;
* scanning;
* score bounds;
* sites;
* reconstruction;
* fixtures;
* compatibility tests;
* benchmarks.

Не создавай общий «универсальный» путь раньше, чем понятны различия моделей.

## Этап 6. Null distributions

Реализовать:

* pair scheduling;
* deterministic parallelism;
* raw score storage;
* GEV fitting;
* p-values;
* format;
* compatibility corpus.

## Этап 7. CLI

Добавить:

* команды;
* JSON schemas;
* logs/progress;
* exit codes;
* compatibility tests.

## Этап 8. Distribution

Подготовить:

* General registry compatibility;
* GitHub releases;
* package app;
* optional PackageCompiler app;
* Linux x86_64;
* Linux aarch64 при возможности;
* macOS arm64;
* conda/Bioconda strategy;
* installation documentation.

# 25. Definition of Done

Перенос считается завершённым только если:

1. Реализован согласованный feature set.
2. Основные Python fixtures воспроизводятся в установленных tolerances.
3. Все расхождения документированы.
4. Публичный API стабилен и документирован.
5. Core library не зависит от Python.
6. Нет обязательной зависимости от pandas/DataFrames.
7. Нет обязательной зависимости от PythonCall.
8. Нет небезопасного основного формата сериализации.
9. Горячие kernels type-stable.
10. Serial и threaded результаты совпадают.
11. CLI выдаёт чистый machine-readable stdout.
12. Есть migration tools для legacy files.
13. Есть benchmark report.
14. Есть architecture document.
15. Есть downstream contract для MotifHORDE.jl.
16. Пакет проходит Aqua и JET checks.
17. Пакет устанавливается в чистом Julia environment.
18. Документация собирается без ошибок.
19. Тесты не зависят от локальных абсолютных путей.
20. Нет скрытых глобальных mutable singletons.

# 26. Запреты

Не делай следующее:

* не переводить файл за файлом в том же порядке;
* не сохранять Python module hierarchy без анализа;
* не делать каждый Python-класс Julia-типом;
* не использовать `Any` как способ быстро завершить перенос;
* не использовать `Dict{Symbol,Any}` вместо предметных типов;
* не использовать строки для dispatch;
* не копировать Numba-specific structure;
* не оборачивать каждый цикл в broadcasting;
* не использовать pandas-подобные таблицы во внутренних kernels;
* не хранить модели только через Julia Serialization;
* не делать PythonCall обязательной runtime dependency;
* не добавлять GPU implementation без benchmark;
* не добавлять macros, generated functions или metaprogramming там, где достаточно обычной функции;
* не использовать global mutable state;
* не запускать parallelism глубоко внутри kernels;
* не менять numerical semantics ради красивого API без документированного решения;
* не считать совпадение нескольких примеров доказательством совместимости;
* не удалять Python-версию до завершения compatibility harness.

# 27. Формат работы и отчётности

Для каждого этапа предоставляй:

1. Краткое описание архитектурного решения.
2. Какие Python-паттерны не были перенесены и почему.
3. Какие Julia-возможности использованы.
4. Какие публичные контракты добавлены или изменены.
5. Какие tests добавлены.
6. Результаты benchmark.
7. Известные расхождения с Python.
8. Риски следующего этапа.

Любое существенное архитектурное решение оформить как ADR:

```text
docs/adr/0001-model-type-hierarchy.md
docs/adr/0002-sequence-representation.md
docs/adr/0003-storage-format.md
docs/adr/0004-parallelism.md
docs/adr/0005-gev-fitting.md
```

ADR должен содержать:

* context;
* decision;
* alternatives;
* consequences;
* migration impact.

# 28. Первый ожидаемый результат

На первом шаге не переписывай весь проект.

Подготовь:

1. Аудит Python MIMOSA.
2. Feature matrix.
3. Предлагаемую архитектуру Mimosa.jl.
4. Проект публичного API.
5. Предложение по model type hierarchy.
6. Решение по sequence representation.
7. Решение по ragged profiles.
8. Решение по storage format.
9. План compatibility fixtures.
10. План benchmark.
11. Risk register.
12. Этапы реализации с зависимостями между ними.

После этого реализуй только первый вертикальный slice:

```text
PWM/PFM parsing
→ sequence encoding
→ PWM reverse complement
→ matrix motif comparison
→ orientation and offset
→ typed result
→ CLI JSON output
```

Этот slice должен:

* быть полностью протестирован;
* иметь Python compatibility fixtures;
* иметь benchmark;
* иметь документацию;
* не содержать временных `Any`;
* не зависеть от Python runtime;
* демонстрировать финальный архитектурный стиль проекта.

Главный критерий успеха — не количество переписанных файлов, а качество нового Julia-дизайна, численная корректность, предсказуемая производительность и пригодность Mimosa.jl как стабильного ядра для будущего MotifHORDE.jl.
