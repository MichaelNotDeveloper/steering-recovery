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
`alpha = [2, 4, 6, 8, 10]`. Для каждой точки выполняется 100 генераций.
Следовательно, базовый конфиг делает `4 × 1 × 5 × 100 = 2000` генераций.

`steering method` содержит:

- policy вмешательства (`every_step`, `once_at_start` или
  `entropy_threshold`);
- опциональный checkpoint post-steering denoiser.

Для `raw/every_step` выход выбранного GPT-2 block на каждом forward изменяется
ровно как `h + αv`. При наличии denoiser сначала вычисляется `h + αv`, затем
результат передаётся в `DenoiserBundle.denoise_steered`. При `alpha=0` hook не
изменяет hidden и denoiser не вызывается: нулевая точка остаётся общей baseline.

GPT-2 Small, post-steering denoiser, Frozen AG News classifier и GPT-2 Large для
SLOR выполняются в `float32`. Перед запуском проверяется provenance
steering-векторов и denoiser checkpoint: старые reduced-precision артефакты
отклоняются с инструкцией по пересчёту.

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

### Dist-1, Dist-2 и Dist-3

Для каждой 40-токенной сгенерированной части n-граммы строятся непосредственно
по GPT-2 token IDs. Dist-N — доля уникальных n-грамм среди всех возможных:

```text
Dist-N = unique generated token n-grams / (40 - N + 1)
```

Значение лежит в `[0, 1]`: более высокая величина означает меньше повторяющихся
n-грамм. Порядки задаются через `metrics.distinct_orders`; текущий канонический
прогон одновременно считает `N = [1, 2, 3]`. Дополнительная language model для
этих метрик не нужна.

### Syntactic Log-Odds Ratio (SLOR)

SLOR считается на GPT-2 BPE-токенах с помощью frozen `gpt2-large`:

```text
SLOR = (Σ log p_gpt2-large(tᵢ | prompt, t<ᵢ) - Σ log p_unigram(tᵢ)) / N
```

В LM передаются prompt и continuation, но в обе суммы входят только 40
сгенерированных токенов. Таким образом, prompt задаёт контекст, но сам не влияет
на длину и числитель метрики. Более высокий SLOR означает, что контекстная модель
предсказывает последовательность лучше независимой unigram-модели.

Unigram-вероятности оцениваются на GPT-2 BPE-токенах из `description` обучающего
split `sh0416/ag_news` с add-one smoothing. Это фиксирует необходимую для
[стандартного определения SLOR](https://gu-clasp.github.io/people/shalom-lappin/papers/lau-clark-lappin2016_cognitive_science.pdf)
частотную базу и не использует benchmark split `test`. Результат сохраняется в
`metrics/slor_unigram.pt` и повторно используется при resume. Корпус и smoothing
настраиваются в `slor.unigram`.

## Доверительные интервалы и графики

Для каждой метрики рассчитывается mean и percentile bootstrap confidence
interval по 100 генерациям. По умолчанию используется 2000 bootstrap resamples
и 95% CI.

Для каждой пары `(vector, method)` создаются четыре отдельных scatter-plot — для
Dist-1, Dist-2, Dist-3 и SLOR:

- ось X — mean target-class probability;
- ось Y — mean соответствующей метрики;
- цвет точки — значение `alpha`;
- error bars и полупрозрачный эллипс вокруг точки — bootstrap CI обеих метрик;
- ось X и оси Dist-N ограничены диапазоном `[0, 1]`, ось SLOR масштабируется по
  данным.

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
python run_steering_benchmarks.py 'alphas=[1,2,4,6,8,10,12]'
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
├── metrics/slor_unigram.pt
├── plots/<method>__<vector>__distinct_1.png
├── plots/<method>__<vector>__distinct_2.png
├── plots/<method>__<vector>__distinct_3.png
├── plots/<method>__<vector>__slor.png
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

Condition JSONL содержит все 100 генераций, token IDs, все метрики, исходную
тему, seed и число вмешательств. Для каждой точки в файлах примеров сохраняется
восемь генераций: по две для исходных тем World, Sports, Business и Sci/Tech.

`examples.html` — автономная интерактивная галерея. В ней можно фильтровать
примеры по `alpha`, steering vector, методу и исходной теме. У каждой генерации
показываются prompt/continuation, основные метрики и раскрываемый полный JSON со
всей сохранённой метаинформацией, включая token IDs, seed, checkpoint, режим и
число вмешательств.

Запуск возобновляемый. Завершённые condition-файлы с совпадающей сигнатурой
пропускаются; незавершённая точка продолжается из `.partial.jsonl`. Новые
метрики можно дописать к сохранённым генерациям без повторной генерации. Frozen
classifier и `gpt2-large` загружаются последовательно после окончания генерации,
чтобы снизить пиковое потребление памяти.
