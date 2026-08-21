# Бенчмарк steering и post-steering denoiser

Бенчмарк сравнивает исходный steering `h ← h + αv` и произвольные
post-steering denoiser на одной генеративной модели. Весь код находится в
`steering_recovery/steering/benchmarking/`, рядом с генераторами векторов.

## Экспериментальная сетка

Одна точка графика задаётся тройкой:

```text
(steering vector, steering method, alpha)
```

По умолчанию используются четыре сохранённых AG News-вектора, метод `raw` и
`alpha = [0, 0.25, 0.5, 1, 2]`. Для каждой точки выполняется 100 генераций.
Следовательно, базовый конфиг делает `4 × 1 × 5 × 100 = 2000` генераций.

`steering method` содержит:

- policy вмешательства (`every_step`, `once_at_start` или
  `entropy_threshold`);
- опциональный checkpoint post-steering denoiser.

Для `raw/every_step` выход выбранного GPT-2 block на каждом forward изменяется
ровно как `h + αv`. При наличии denoiser сначала вычисляется `h + αv`, затем
результат передаётся в `DenoiserBundle.denoise_steered`. При `alpha=0` hook не
изменяет hidden и denoiser не вызывается: нулевая точка остаётся общей baseline.

GPT-2 Small, post-steering denoiser и Frozen AG News classifier выполняются в
`float32`. Перед запуском проверяется provenance steering-векторов и denoiser
checkpoint: старые reduced-precision артефакты отклоняются с инструкцией по
пересчёту.

## Промпты и генерация

Один детерминированно перемешанный набор промптов используется во всех точках:

1. загружается held-out split `test` датасета `sh0416/ag_news`;
2. случайно выбирается по 25 статей каждой исходной темы — всего 100;
3. из `description` берутся ровно первые 24 GPT-2 token IDs;
4. GPT-2 генерирует ровно 40 новых токенов с KV-cache;
5. один и тот же sample использует одинаковый seed при всех методах, векторах и
   значениях `alpha`.

Стратификация исходных статей не связывает их тему с направлением steering, но
гарантирует, что примеры и средние не зависят от случайного перекоса тем.
`stop_on_eos=false` оставлен по умолчанию, чтобы каждая генерация содержала
ровно 40 токенов даже при сэмплировании EOS.

## Метрики

### Target-class probability

Только декодированная 40-токенная сгенерированная часть передаётся в frozen
[`mansoorhamidzadeh/ag-news-bert-classification`](https://huggingface.co/mansoorhamidzadeh/ag-news-bert-classification).
Из softmax берётся вероятность класса, соответствующего текущему steering
vector:

```text
World=0, Sports=1, Business=2, Sci/Tech=3
```

Соответствие задано явно в `classifier.class_indices` и не зависит от текстовых
значений `id2label` конкретной версии checkpoint.

### Dist-3

Для каждой 40-токенной сгенерированной части строятся триграммы непосредственно
по GPT-2 token IDs. Dist-3 — доля уникальных триграмм среди всех возможных:

```text
Dist-3 = unique generated token trigrams / 38 generated token trigrams
```

Значение лежит в `[0, 1]`: более высокая величина означает меньше повторяющихся
триграмм. Порядок `n` задаётся через `metrics.distinct_n`; текущий канонический
прогон использует `n=3`. Дополнительная language model для этой метрики не нужна.

## Доверительные интервалы и графики

Для каждой метрики рассчитывается mean и percentile bootstrap confidence
interval по 100 генерациям. По умолчанию используется 2000 bootstrap resamples
и 95% CI.

Для каждой пары `(vector, method)` создаётся отдельный scatter-plot:

- ось X — mean target-class probability;
- ось Y — mean Dist-3;
- цвет точки — значение `alpha`;
- error bars и полупрозрачный эллипс вокруг точки — bootstrap CI обеих метрик;
- обе оси ограничены диапазоном `[0, 1]`.

## Методы и denoiser

Базовый метод находится в `configs/steering_benchmark.yaml`:

```yaml
methods:
  - name: raw
    intervention_mode: every_step
    denoiser_checkpoint: null
```

Несколько post-steering denoiser добавляются в тот же список:

```yaml
methods:
  - name: raw
    intervention_mode: every_step
    denoiser_checkpoint: null
  - name: denoiser_sigma_0p2
    intervention_mode: every_step
    denoiser_checkpoint: /path/to/latent_3072_layers_5_sigma_0p2/best.pt
  - name: denoiser_sigma_0p5
    intervention_mode: every_step
    denoiser_checkpoint: /path/to/latent_3072_layers_5_sigma_0p5/best.pt
```

Имя метода должно быть уникальным и filesystem-safe. Checkpoint загружается
существующим `load_checkpoint` и должен иметь `format_version=2`.

## Запуск

Полный базовый прогон:

```bash
python run_steering_benchmarks.py
```

Изменение сетки `alpha`:

```bash
python run_steering_benchmarks.py 'alphas=[0,0.1,0.25,0.5,1,2,4]'
```

Smoke-прогон допустим для проверки окружения, но не является каноническим
результатом на 100 генерациях:

```bash
python run_steering_benchmarks.py \
  generation.samples_per_point=8 \
  statistics.bootstrap_resamples=100 \
  hydra.run.dir=runs/steering-benchmarks/smoke
```

`samples_per_point` должен делиться на четыре, поскольку prompts выбираются
поровну по исходным темам.

## Артефакты запуска

```text
runs/steering-benchmarks/<date>/<time>/
├── conditions/<method>/<vector>/alpha_<value>.jsonl
├── plots/<method>__<vector>.png
├── prompts.jsonl
├── prompts_metadata.json
├── summary.csv
├── summary.jsonl
├── examples.jsonl
├── examples.md
├── examples.html
├── config.yaml
└── manifest.json
```

Condition JSONL содержит все 100 генераций, token IDs, обе метрики, исходную
тему, seed и число вмешательств. Для каждой точки в файлах примеров сохраняется
восемь генераций: по две для исходных тем World, Sports, Business и Sci/Tech.

`examples.html` — автономная интерактивная галерея. В ней можно фильтровать
примеры по `alpha`, steering vector, методу и исходной теме. У каждой генерации
показываются prompt/continuation, основные метрики и раскрываемый полный JSON со
всей сохранённой метаинформацией, включая token IDs, seed, checkpoint, режим и
число вмешательств.

Запуск возобновляемый. Завершённые condition-файлы с совпадающей сигнатурой
пропускаются; незавершённая точка продолжается из `.partial.jsonl`. Frozen
classifier загружается только после окончания генерации, чтобы снизить пиковое
потребление памяти.
