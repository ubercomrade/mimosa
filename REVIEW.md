# Design Review by `coding-standards`

Дата проверки: 2026-06-10.

Область проверки: `src/mimosa/*.py`, фокус на Design rules из skill `coding-standards`:

- `design-philosophy`
- `design-single-responsibility`
- `design-dependency-injection`
- `design-pure-functions`
- `design-functional-core-shell`
- `design-explicit-data-flow`
- `design-immutable-config-results`
- `design-functional-pipeline`
- `design-singledispatch`
- `design-avoid-functional-overabstraction`
- `design-early-return`

Проверка была статической: код и публичный API прочитаны, тесты не запускались.

## Краткий вывод

Проект в целом близок к правилам skill: много чистых функций, явный registry-based dispatch для моделей, численные kernels отделены от Python orchestration, CLI в основном держится на краю системы. Основные проблемы не в алгоритмах, а в публичной модели данных и границах слоев:

1. Публичные имена `ComparisonConfig` и `OneToManyConfig` несимметричны: первое фактически означает `1 vs 1`, второе - `1 vs many`.
2. Config/result объекты сделаны как mutable `TypedDict`, хотя skill требует immutable config/result objects.
3. `comparison.py` смешивает core comparison, cache access и significance/null annotation.
4. `models.py` и `comparison.py` стали слишком крупными многоответственными модулями.
5. Есть несколько реальных повторов логики, но не все похожие фрагменты стоит объединять.

## Оценка по Design rules

| Rule | Статус | Комментарий |
| --- | --- | --- |
| `design-philosophy` | Частично | Простые функции используются хорошо, но публичные имена и крупные модули усложняют модель проекта. |
| `design-single-responsibility` | Частично | `comparison.py`, `models.py`, `cli.py`, `nulls.py` имеют несколько причин для изменения. |
| `design-dependency-injection` | Частично | Registry-подход нормален, но null/cache I/O зашит в comparison path. |
| `design-pure-functions` | Частично | Numerical core в основном чистый; result annotation мутирует dict in place. |
| `design-functional-core-shell` | Частично | CLI является shell, но `comparison.compare()` уже вызывает null-artifact resolution. |
| `design-explicit-data-flow` | Частично | Большинство параметров явные, но generic `dict`/`TypedDict` и ключи вроде `model1`/`model2` скрывают семантику. |
| `design-immutable-config-results` | Не соответствует | Public configs/results mutable dict-like objects. |
| `design-functional-pipeline` | Частично | Pipelines читаемые, но CLI/null orchestration можно сделать более явным набором этапов. |
| `design-singledispatch` | Соответствует с оговоркой | Для моделей используется registry по `type_key`; это уместнее, чем `singledispatch`, потому что dispatch идет по строковому формату модели. |
| `design-avoid-functional-overabstraction` | Соответствует | Не видно monad/Result/Maybe-style overabstraction. |
| `design-early-return` | В основном соответствует | Guard clauses используются часто; отдельные CLI ветки можно еще упростить, но это не главный риск. |

## Наиболее важные замечания

### [x] 1. `ComparisonConfig` должен быть переименован в `OneToOneConfig`

Файл: `src/mimosa/api.py:45`, `src/mimosa/api.py:61`, `src/mimosa/__init__.py:3`.

Сейчас:

- `ComparisonConfig` описывает сравнение `model1` vs `model2`.
- `OneToManyConfig` описывает `query` vs `targets`.

Это нарушает однозначность имен: `ComparisonConfig` звучит как общий config для любого сравнения, хотя на практике это `1 vs 1`.

Рекомендуемая схема без обратной совместимости:

- `ComparisonConfig` -> `OneToOneConfig`
- `create_config()` -> `create_one_to_one_config()`
- `run_comparison()` -> `run_one_to_one()`
- `create_many_config()` -> `create_one_to_many_config()`
- `compare_motifs()` можно оставить как high-level shortcut, но внутри он должен использовать `OneToOneConfig`.

Старые имена лучше удалить, а не оставлять aliases. Это даст более чистую кодовую базу и не позволит двум наборам имен жить параллельно:

```python
@dataclass(frozen=True, slots=True)
class OneToOneConfig:
    query: ModelRef
    target: ModelRef
    query_type: str | None
    target_type: str | None
    ...
```

Тесты и импорты нужно переводить сразу на новые имена: `tests/unit_comparison.py:1246`, `tests/unit_comparison.py:1265`, `tests/unit_comparison.py:1285`, `tests/unit_comparison.py:1321` и далее API shortcut tests.

### [x] 2. Не все config/result типы должны жить в `api.py`, но публичные contracts должны иметь один источник истины

Файлы:

- `src/mimosa/api.py:45` - `ComparisonConfig`
- `src/mimosa/api.py:61` - `OneToManyConfig`
- `src/mimosa/comparison.py:52` - `ComparatorConfig`
- `src/mimosa/comparison.py:72` - `ComparisonResult`
- `src/mimosa/batches.py:17` - `SequenceBatch`
- `src/mimosa/cache.py:20` - `ProfileCacheSpec`
- `src/mimosa/nulls.py:42` - null artifact typed dicts

В текущем коде нет второго определения `ComparisonConfig`: он определен в `api.py`, а `__init__.py` только реэкспортирует. Это не DRY-проблема.

Но есть другая проблема: публичные contracts разбросаны по слоям. Например `ComparisonResult` объявлен в `comparison.py`, но используется как public API result в `api.py`. Если требование проекта - "публичные config/result классы только в API", тогда стоит:

- держать `OneToOneConfig`, `OneToManyConfig`, `ComparatorConfig`, `ComparisonResult` в `api.py` или в отдельном `types.py`;
- в `comparison.py` импортировать эти типы, а не объявлять публичные типы внутри implementation module;
- оставить module-local specs там, где они не часть public API: `ProfileCacheSpec` в `cache.py`, `NullArtifact*` в `nulls.py`, `SequenceBatch` в `batches.py`.

Если цель именно "все публичные типы из одного места", лучше завести `src/mimosa/types.py`. Если цель "API является публичной точкой входа", тогда можно определить public contracts в `api.py` и реэкспортировать их из `__init__.py`.

### [x] 3. Config/result объекты сейчас mutable, хотя rule требует immutable config/result objects

Файлы:

- `src/mimosa/api.py:45`
- `src/mimosa/api.py:61`
- `src/mimosa/comparison.py:52`
- `src/mimosa/comparison.py:72`
- `src/mimosa/models.py:58`
- `src/mimosa/nulls.py:74`

Сейчас основные contracts сделаны как `TypedDict`. Это удобно для JSON-like payloads, но:

- config можно случайно изменить после создания;
- result annotation добавляет поля позже;
- schema не защищает от частичных/лишних ключей в runtime;
- именованные поля не документируют доменную семантику так хорошо, как dataclass.

Особенно заметные места:

- `create_comparator_config()` собирает mutable dict: `src/mimosa/comparison.py:120`.
- `_copy_comparator_config()` копирует dict, чтобы безопасно мутировать background: `src/mimosa/comparison.py:166`.
- `annotate_results_with_nulls()` изменяет result dict in place: `src/mimosa/nulls.py:542`.
- `GenericModel.config` - обычный mutable `dict`: `src/mimosa/models.py:66`.

Рекомендуемая цель:

- `@dataclass(frozen=True)` для `OneToOneConfig`, `OneToManyConfig`, `ComparatorConfig`, `ComparisonResult`.
- Для optional significance fields либо отдельный `AnnotatedComparisonResult`, либо `ComparisonResult` с optional полями и function `with_null_annotations(...) -> list[ComparisonResult]`, которая возвращает новые объекты.
- `GenericModel.config` заменить на typed immutable metadata, например `Mapping[str, Any]` как минимум, лучше отдельные model-specific config dataclasses позже.

Это самая важная design-правка после переименования API.

### [x] 4. `comparison.py` смешивает core comparison и null-significance I/O

Файлы:

- `src/mimosa/comparison.py:1072`
- `src/mimosa/comparison.py:1103`
- `src/mimosa/comparison.py:1144`
- `src/mimosa/nulls.py:429`

`compare()` и `compare_one_to_many()` должны быть core comparison entry points. Но сейчас они:

1. считают score;
2. копируют и модифицируют config;
3. вызывают `_maybe_annotate_significance()`;
4. `_maybe_annotate_significance()` импортирует `mimosa.nulls`;
5. `nulls` грузит joblib artifacts и ищет их в cache/search dirs.

Это нарушает `design-functional-core-shell` и `design-single-responsibility`: score comparison зависит от artifact loading policy.

Рекомендуемый split:

- `comparison.compare_pair(...) -> ComparisonResult` - только score.
- `comparison.compare_one_to_many(...) -> list[ComparisonResult]` - только score.
- `api.compare_motifs(..., pvalue=True)` или отдельный orchestration layer решает, нужно ли annotating.
- `nulls.annotate_results_with_nulls(...)` лучше сделать pure-ish: возвращать новые results, а не мутировать входной список.
- Artifact loading/search оставить на API/CLI boundary.

Так comparison core станет проще тестировать и переиспользовать без файловой системы.

### [x] 5. Повтор генерации случайных последовательностей нужно вынести

Файлы:

- `src/mimosa/api.py:374`
- `src/mimosa/cli.py:602`

Обе функции делают один и тот же доменный шаг:

- validate `num_sequences`;
- validate `seq_length`;
- create `np.random.default_rng(seed)`;
- generate integer A/C/G/T rows;
- return `make_sequence_batch(rows)`.

Это настоящий DRY-случай: один и тот же knowledge повторяется в API и CLI shell.

Рекомендуемое место:

- `batches.py`: `make_random_sequence_batch(num_sequences: int, seq_length: int, seed: int) -> SequenceBatch`

После этого:

- `api._generate_random_sequences()` можно удалить;
- `cli._resolve_build_null_sequences()` будет вызывать общий helper.

### [ ] 6. `models.py` стал модулем с несколькими responsibilities

Файл: `src/mimosa/models.py`, 935 строк.

Сейчас в одном модуле находятся:

- public model registry: `GenericModel`, `ModelHandler`, `register_model_handler`;
- read/write dispatch;
- scan dispatch;
- threshold table helpers;
- hit extraction;
- site table creation;
- PFM reconstruction;
- concrete model loaders/scanners for PWM/SiteGA/BaMM/Dimont/Slim/scores.

Это работает, но нарушает `design-single-responsibility`: разные изменения будут часто попадать в один файл.

Рекомендуемый split без изменения поведения. Для новых файлов лучше использовать однословные имена без `_`; если логика естественно ложится в существующий файл, лучше перенести туда, а не плодить новый модуль.

- `models.py` - public facade and registry.
- `handlers.py` - concrete registrations/loaders/scanners.
- `scanning.py` - scan dispatch and score bounds.
- `sites.py` - `get_sites`, hit extraction, PFM reconstruction.

Такой split не обязан менять public API: `models.py` может продолжить реэкспортировать функции.

### [ ] 7. `comparison.py` тоже стоит разделить по стратегиям

## Выполнено по факту

- Переименован публичный `1 vs 1` API: `OneToOneConfig`, `create_one_to_one_config()`, `run_one_to_one()`.
- Публичные immutable contracts вынесены в единый модуль [`src/mimosa/types.py`](src/mimosa/types.py).
- `ComparatorConfig`, `ComparisonResult`, `OneToOneConfig`, `OneToManyConfig` переведены на `@dataclass(frozen=True, slots=True)`.
- P-value/null annotation вынесен с comparison core на API boundary; `nulls.annotate_results_with_nulls()` теперь возвращает новые результаты вместо in-place mutation.
- Генерация случайных sequence batch вынесена в `batches.make_random_sequence_batch()`, дубли из `api.py` и `cli.py` удалены.

Файл: `src/mimosa/comparison.py`, 1177 строк.

Сейчас здесь:

- config creation and validation;
- metric constants;
- profile normalization and cache coordination;
- profile alignment;
- motif tensor normalization and motif alignment;
- one-to-many joblib parallelization;
- null annotation hook.

Минимальный split. Новые имена также лучше держать однословными:

- `comparison.py` - public facade, registry, common config validation.
- `profile.py` - profile strategy.
- `motif.py` - motif strategy.
- `results.py` или `types.py` - result/config contracts and result construction.

Это поддерживает `design-functional-pipeline`: load/prepare/score/annotate stages станут виднее.

### 8. CLI содержит слишком много orchestration для null building

Файл: `src/mimosa/cli.py:544`.

`run_build_null_from_args()` делает сразу:

- load models;
- collect relation inputs;
- parse relations;
- build comparator;
- resolve sequences/background;
- build null distributions;
- save artifact;
- optionally install cache;
- print summary.

Это нормальная imperative shell, но функция уже слишком широкая. Лучше выделить pure-ish builder:

- `build_null_request_from_args(args) -> NullBuildRequest`
- `run_build_null_request(request) -> NullBuildSummary`
- CLI оставить только parse/print/exit.

Это снизит размер CLI и упростит тесты без subprocess.

### 9. Result annotation мутирует результаты in place

Файл: `src/mimosa/nulls.py:542`.

`annotate_results_with_nulls()` меняет `results` на месте через `result.update(...)` и `results[idx]["q-value"] = ...`.

Проблемы:

- side effect не очевиден из имени;
- `ComparisonResult` в `comparison.py:72` не содержит p-value/E-value/q-value/null fields;
- public result schema фактически шире, чем declared schema.

Рекомендация:

- сделать `annotate_results_with_nulls(...) -> list[ComparisonResult]`;
- возвращать новые result objects;
- обновить schema результата, добавив optional fields или отдельный `SignificantComparisonResult`.

### 10. `GenericModel` выглядит как data container, но не immutable

Файл: `src/mimosa/models.py:58`.

`GenericModel` - хороший кандидат на immutable result/config-like object:

```python
@dataclass(frozen=True, slots=True)
class GenericModel:
    type_key: str
    name: str
    representation: Any
    length: int
    config: Mapping[str, Any]
```

Оговорка: `np.ndarray` внутри останется mutable, даже при `frozen=True`. Но frozen dataclass хотя бы запретит переназначение полей и сделает intent яснее.

### 11. Registry вместо `singledispatch` здесь оправдан, но регистрацию моделей стоит сделать явнее

Файл: `src/mimosa/models.py:77`, `src/mimosa/comparison.py:95`.

Skill рекомендует `singledispatch`, когда dispatch зависит от типа первого аргумента. Здесь dispatch зависит от строкового `type_key`/strategy name и нужен registry для file-format handlers. Это не нарушение.

Что стоит улучшить:

- заменить raw `TypedDict` handler на immutable dataclass, чтобы handler выглядел как контракт, а не произвольный dict;
- сделать `DEFAULT_MODEL_HANDLERS` явным словарем/tuple рядом с регистрацией, чтобы список поддерживаемых моделей был виден в одном месте;
- оставить public `register_model_handler()` только для расширений, но использовать его через один понятный validation path;
- убрать два похожих уровня `_register_model_handler()` и `register_model_handler()`, если приватный слой не добавляет отдельной логики;
- назвать registry точнее, например `MODEL_HANDLERS`, если он остается module-level объектом;
- добавить `available_model_types()` или `supported_model_types()`, чтобы CLI/API не дублировали списки типов вручную;
- при split в `handlers.py` держать concrete handler construction рядом с конкретными `_load_*`, `_scan_*`, `_write_*` функциями, а в `models.py` оставить только lookup/registration facade.

Целевая форма:

```python
@dataclass(frozen=True, slots=True)
class ModelHandler:
    scan: ScanFunction
    scan_both: ScanBothFunction | None
    load: LoadFunction
    write: WriteFunction
    score_bounds: ScoreBoundsFunction


MODEL_HANDLERS: dict[str, ModelHandler] = {
    "pwm": ModelHandler(...),
    "sitega": ModelHandler(...),
    "bamm": ModelHandler(...),
    "dimont": ModelHandler(...),
    "slim": ModelHandler(...),
    "scores": ModelHandler(...),
}
```

Так регистрация становится декларативной: читатель сразу видит список форматов и функции, которые обслуживают каждый формат.

### 12. Не надо механически убирать все похожие повторы

По `design-philosophy`, DRY применяется к знанию, а не к любому похожему тексту.

Примеры повторов, которые пока допустимы:

- `SequenceBatch`, `MaskedBatch`, `ProfileBundle` в `batches.py` и `ProfileCacheSpec` в `cache.py` описывают разные contracts. Их не нужно тащить в `api.py`.
- `_build_profile_result()` и `_build_motif_result()` похожи, но у profile есть `n_sites`, у motif нет. Можно объединить позже, если появится общий `ComparisonResult` dataclass.
- `_scan_pwm`, `_scan_sitega`, `_scan_bamm`, `_scan_dimont`, `_scan_slim` похожи как registry adapters, но это осознанный thin adapter layer.

Примеры повторов, которые стоит убрать:

- random sequence generation in `api.py` and `cli.py`;
- public `model1/model2` terminology for one-to-one while `query/target` is used elsewhere;
- result schema split between base result and significance-enriched result.

## Рекомендуемый порядок исправлений

1. Ввести `OneToOneConfig` и `create_one_to_one_config()`; удалить `ComparisonConfig` и `create_config()`.
2. Переименовать `create_many_config()` в `create_one_to_many_config()` и удалить старое имя.
3. Переименовать `run_comparison()` в `run_one_to_one()`; удалить старое имя.
4. Согласовать public terminology:
   - one-to-one config: `query`, `target`, `query_type`, `target_type`;
   - one-to-many config: `query`, `targets`, `query_type`, `target_type`;
   - не использовать `model1/model2` в новых public API names.
5. Перевести public configs/results с `TypedDict` на frozen dataclasses.
6. Вынести random sequence generation в `batches.make_random_sequence_batch()`.
7. Сделать регистрацию моделей декларативной: `ModelHandler` dataclass + `MODEL_HANDLERS` + `supported_model_types()`.
8. Убрать null annotation из `comparison.compare()` и перенести ее в API/CLI orchestration.
9. Сделать `annotate_results_with_nulls()` возвращающей новые results.
10. Разделить крупные файлы только там, где это снижает responsibility; новые модули называть одним словом:
    - `handlers.py`, если нужно вынести concrete model handlers;
    - `scanning.py`, если нужно отделить scan dispatch;
    - `sites.py`, если нужно отделить sites/PFM reconstruction;
    - `profile.py`, `motif.py`, `results.py` или `types.py`, если нужно разделить `comparison.py`.

## Что поправить в первую очередь

Самый практичный первый PR:

- добавить `OneToOneConfig`;
- добавить `create_one_to_one_config`;
- добавить `create_one_to_many_config`;
- добавить `run_one_to_one`;
- перевести поля `model1/model2` на `query/target`;
- обновить `__init__.py`;
- удалить старые имена `ComparisonConfig`, `create_config`, `create_many_config`, `run_comparison`;
- обновить все тесты и внутренние импорты под новые имена.

Это решает главную проблему однозначности имен без большого риска для алгоритмов и без сохранения устаревшего API.
