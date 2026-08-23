# Steering Recovery

Воспроизводимый пайплайн для экспериментов с activation steering:

- streaming teacher-forced hidden states из OpenWebText через GPT-2;
- опциональное кеширование hidden states в `.npy`;
- baseline-генерация без вмешательства и с режимами `once_at_start`,
  `every_step`, `entropy_threshold`;
- совместное обучение grid residual MLP denoiser на Gaussian noise;
- Hydra-конфигурации и multirun-перебор гиперпараметров;
- метрики и артефакты в Weights & Biases;
- расширяемая генерация steering-векторов по размеченным датасетам.

Все пути с GPT-2 Small работают только в `float32`: конфиги задают FP32 явно,
а код отклоняет `fp16`/`bf16` и несовместимые старые артефакты. Порядок
пересчёта результатов описан в [политике точности](docs/precision.md).

## Быстрый старт

```bash
conda env create -f environment.yaml
conda activate steering-recovery
```

Сначала соберите статистики GPT-2 hidden states. Лимит считается по реально
добавленным hidden-токенам (padding и первый токен каждого текста не входят):

```bash
python collect_hidden_statistics.py \
  source.layer_index=5 \
  collection.max_tokens=1000000 \
  output_path=data/gpt2_layer_5_statistics.pt
```

Сбор идёт с `tqdm`; промежуточные моменты считаются в `float64` алгоритмом
Chan/Welford без хранения активаций, а сам GPT-2 forward выполняется в
`float32`. Затем запустите обучение:

```bash
python train_denoiser.py \
  data.streaming.model_name=gpt2 \
  data.streaming.layer_index=5 \
  data.statistics_path=data/gpt2_layer_5_statistics.pt \
  training.batch_size=512 \
  training.max_steps=10000
```

`training.batch_size` — точное число hidden states, которое `IterableDataset`
выдаёт за одну итерацию. Неполный остаток переносится между текстами.
На каждом batch обучаются все 27 комбинаций
`latent_dim=[192,768,3072] × num_layers=[1,3,5] × sigma=[0.1,0.2,0.5]`.
Без корректного `data.statistics_path` обучение завершится с ошибкой до первого
optimizer step.

## Формат статистик hidden states

Файл сохраняется через `torch.save` как словарь:

```python
{
    "format_version": 1,
    "sum": Tensor[hidden_size],       # float64
    "variance": Tensor[hidden_size],  # float64, population variance
    "count": int,                     # число hidden-токенов
    "source": {...},                  # модель, слой и max_length
    "dataset": {...},                 # источник текстов
}
```

`source` также содержит фактический `model_dtype`; для GPT-2 Small допустимо
только значение `float32`.

Два вектора должны быть одномерными, одинакового размера и соответствовать
тому же GPT-2 layer, который используется при обучении. Denoiser вычисляет
`mean = sum / count` и `std = sqrt(variance)`, после чего сохраняет их внутри
checkpoint в состоянии нормализатора. `count` — обязательный скаляр: без него
невозможно восстановить mean из sum.

Опциональный статический режим:

```bash
python train_denoiser.py \
  data.mode=static \
  data.path=/path/to/activations \
  data.statistics_path=/path/to/gpt2_layer_5_statistics.pt
```

Сводная таблица и barplot после обучения:

```bash
python compare_denoisers.py /path/to/run --output-dir comparison
```

Скрипт сохраняет общий barplot и отдельный PNG для каждого
`sigma`; все оси метрик логарифмические. Новые запуски также отображают
`score_rms = sqrt(mean((denoised - noisy)²) / sigma⁴)`. Тонкие чёрные ticks
показывают identity-baseline `f(y)=y` для каждой модели.

Baseline без steering:

```bash
python run_baselines.py \
  data.prompts_path=/path/to/prompts.jsonl \
  steering.mode=once_at_start steering.scale=0 \
  wandb.enabled=false
```

Steering-векторы четырёх тем AG News (`World`, `Sports`, `Business`,
`Sci/Tech`) по hidden-состоянию шестого блока GPT-2:

```bash
python generate_steering_vectors.py
```

Команда обрабатывает по 1000 статей каждой темы и сохраняет отдельные векторы и
полные метаданные в `data/steering_vectors/ag_news/gpt2_layer_5/`. Difference of
Means использует hidden всех токенов полного текста. Формат, старый prompt-режим
и расширение пайплайна описаны в
[документации по steering-векторам](docs/steering_vectors.md).

Отдельное обучение четырёх one-vs-rest L2 logistic regression по большой
сбалансированной выборке hidden-состояний:

```bash
python train_topic_logistic_regressions.py
```

Запуск сохраняет loss, ROC-AUC, AUC-PRC по эпохам и интерактивный HTML с четырьмя
режимами токенной подсветки. Подробности: [классификаторы тем
AG News](docs/topic_logistic_regression.md).

Бенчмарк силы steering и post-steering denoiser:

```bash
python run_steering_benchmarks.py
```

Для каждой пары «AG News-вектор × метод» строится scatter-график target-class
probability против Dist-3 с интерактивной HTML-галереей примеров. Цвет кодирует
`alpha`, а вокруг точек показывается 95% bootstrap CI. Подробности и формат результатов:
[документация по бенчмаркам](docs/steering_benchmarks.md).

MC-dropout эксперимент необычности steered hidden states:

```bash
python train_denoiser.py experiment=epistemic_dropout
python run_epistemic_steering.py denoiser_run_dir=/path/to/training/run
```

Первый этап обучает три `3 × 3072` denoiser с `dropout=0.1` для
`sigma=[0.1,0.2,0.5]`. Второй прогоняет каждый steered hidden 20 раз, сохраняет
восемь мер разброса и геометрии steering, строит графики и интерактивную
токенную HTML-галерею.
Подробности: [epistemic steering](docs/epistemic_steering.md).

Все параметры можно переопределять из CLI. Подробности: [архитектура](docs/architecture.md),
[обучение denoiser](docs/denoiser_training.md) и
[генерация steering-векторов](docs/steering_vectors.md),
[классификаторы тем](docs/topic_logistic_regression.md),
[бенчмарки steering](docs/steering_benchmarks.md),
[MC-dropout epistemic steering](docs/epistemic_steering.md).

## Проверки

```bash
pytest
```
