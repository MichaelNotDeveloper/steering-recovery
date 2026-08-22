# Обучение denoiser

## Задача

Каждый GPT hidden state сначала нормализуется фиксированными статистиками:

```text
x = (h - mean) / std
x_noisy = x + sigma * epsilon,  epsilon ~ N(0, I)
x_recovered = denoiser(x_noisy)
loss = mean((x_recovered - x)²)
```

GPT-2 Small, его hidden states и весь train/validation denoiser работают в
`float32`. Autocast отключён (`training.precision=fp32`). Статистики без
`source.model_dtype=float32` считаются несовместимыми и должны быть собраны
заново.

Модель предсказывает чистый нормализованный hidden state напрямую. Для каждого
`sigma` обучается отдельная модель; noise level не передаётся на вход сети.

## Residual MLP

Один блок не содержит внутренней нормализации:

```text
residual = Linear(hidden_size -> latent_dim, bias=True)
residual = GELU(residual)
residual = Linear(latent_dim -> hidden_size, bias=True)
residual = Dropout(p=dropout)(residual)
output = input + residual
```

`model.dropout` задаёт dropout после второго Linear каждого residual block и по
умолчанию равен `0.0`, поэтому прежняя архитектура и checkpoints остаются
совместимыми. Dropout активен при обучении, отключён на обычной validation и
может быть точечно включён для MC-dropout inference.

`hidden_size=768` задаётся GPT-2 и не является параметром grid. Внутренняя
размерность `latent_dim` перебирается как `[192, 768, 3072]`. Полная сетка:

- `latent_dim`: `192`, `768`, `3072`;
- число residual blocks: `1`, `3`, `5`;
- `sigma`: `0.1`, `0.2`, `0.5`.

Всего одновременно обучаются 27 независимых моделей. Они находятся в памяти
одного процесса и обновляются lockstep: train loop получает один новый batch,
нормализует его один раз, создаёт один `epsilon`, после чего каждая модель
получает этот же batch и `x + sigma * epsilon`. Так сравнение архитектур не
зависит от разных текстов или разных реализаций Gaussian noise. Одинаковые
архитектуры для разных `sigma` также получают одинаковую начальную инициализацию.

Отдельный preset для epistemic-эксперимента оставляет только три модели:

```bash
python train_denoiser.py experiment=epistemic_dropout
```

Все три имеют `latent_dim=3072`, три residual block и `dropout=0.1`; различается
только `sigma`: `0.1`, `0.2`, `0.5`. Подробный второй этап эксперимента описан в
[epistemic steering benchmark](epistemic_steering.md).

## Подготовка и запуск

Сначала соберите статистики того же GPT-2 layer:

```bash
python collect_hidden_statistics.py \
  source.model_name=gpt2 \
  source.model_dtype=float32 \
  source.layer_index=5 \
  collection.max_tokens=1000000 \
  output_path=data/gpt2_layer_5_statistics.pt
```

Запуск полного grid:

```bash
python train_denoiser.py \
  data.streaming.model_name=gpt2 \
  data.streaming.model_dtype=float32 \
  data.streaming.layer_index=5 \
  data.statistics_path=data/gpt2_layer_5_statistics.pt \
  training.batch_size=512 \
  training.max_steps=10000 \
  training.validation_every_batches=100 \
  training.validation_batches=50 \
  training.precision=fp32
```

Validation запускается каждые `validation_every_batches` train-батчей и ещё раз
в конце обучения. Для всех моделей используются одинаковые validation batches и
одинаковый детерминированный `epsilon`.

Для небольшого smoke run можно сократить grid:

```bash
python train_denoiser.py \
  data.mode=static \
  data.path=/data/small \
  data.statistics_path=/data/small-statistics.pt \
  'model.latent_dims=[192]' \
  'model.num_layers=[1]' \
  'model.sigmas=[0.1]' \
  training.max_steps=20 training.batch_size=16 \
  training.validation_every_batches=5 \
  training.precision=fp32 device=cpu wandb.enabled=false
```

## Метрики

Для noisy input и результата модели считаются:

- `l2` — MSE между восстановленным и чистым нормализованным hidden;
- `rmse` — квадратный корень из `l2`;
- `cosine_distance` — `1 - cosine_similarity(recovered, clean)`;
- `noisy_l2`, `noisy_rmse`, `noisy_cosine_distance` — те же baseline-метрики
  между зашумлённым и чистым hidden;
- `score_mse` — средний квадрат прямой оценки score:
  `mean((denoised - noisy)²) / sigma⁴`;
- `score_rms` — `sqrt(score_mse)`, то есть RMS-компонента оценки
  `nabla log p_sigma(noisy)`.

Для `y = x + sigma * epsilon` формула Tweedie имеет вид:

```text
E[x | y] = y + sigma² * nabla_y log p_sigma(y)
score_hat(y) = (denoiser(y) - y) / sigma²
score_rms = sqrt(mean((denoiser(y) - y)²) / sigma⁴)
```

Нормировать на `sigma⁴` нужно квадрат смещения `denoiser(y) - y`, а не
reconstruction loss `mean((x - denoiser(y))²)`. В оптимуме покомпонентный
reconstruction loss связан со score так:

```text
loss_opt = sigma² - sigma⁴ * E[score(y)²]
```

После деления смещения на `sigma²` явный масштаб шума сокращается, но
`p_sigma` — распределение, сглаженное шумом данного уровня, поэтому значение
score всё равно может зависеть от `sigma`. Для score-метрик требуется
`sigma > 0`.

Лучший checkpoint каждой модели выбирается по минимальному validation `l2`.
W&B получает метрики с namespace:

```text
models/<variant>/train/l2
models/<variant>/val/l2
models/<variant>/val/rmse
models/<variant>/val/cosine_distance
models/<variant>/val/score_mse
models/<variant>/val/score_rms
```

## Результаты

Структура одного запуска:

```text
run/
├── config.yaml
├── statistics.pt
├── grid_summary.json
└── models/
    └── latent_192_layers_1_sigma_0p1/
        ├── model_config.json
        ├── metrics.jsonl
        ├── summary.json
        ├── best.pt
        └── last.pt
```

`best.pt` содержит лучшие веса по validation L2. `last.pt` дополнительно
содержит optimizer state последнего шага. `metrics.jsonl` хранит всю историю
train/validation, а `summary.json` — параметры и лучшие validation-метрики.
Параметр `dropout` входит в checkpoint, `model_config.json`, summary и таблицу
`compare_denoisers.py`; для старых checkpoint/summary без этого поля принимается
`dropout=0`.
Новый прямой denoiser использует checkpoint format version 2; checkpoint старой
модели, которая предсказывала corruption, с ним несовместим.

## Таблица и barplot

Скрипт рекурсивно проходит по всем `summary.json` под заданной директорией:

```bash
python compare_denoisers.py runs/denoiser/2026-08-20/12-00-00 \
  --output-dir comparison
```

Он создаёт:

- `denoiser_comparison.csv`;
- `denoiser_comparison.md`;
- `denoiser_comparison.png` — горизонтальные barplot для L2, RMSE, cosine
  distance и score RMS, отсортированные по лучшему validation L2;
- `denoiser_comparison_sigma_<sigma>.png` — отдельный график для
  каждого уровня шума.

Во всех PNG оси L2, RMSE и cosine distance используют
логарифмическую шкалу. Score RMS также отображается логарифмически. Старые
`summary.json`, созданные до добавления `score_mse`/`score_rms`, по-прежнему
читаются, но score-панель для них недоступна: эту величину нельзя точно
восстановить только из reconstruction L2.

Тонкий чёрный tick поверх каждого bar показывает результат identity-baseline
`f(y) = y`, то есть модели без денойзинга. Для L2, RMSE и cosine distance это
сохранённые `noisy_*` метрики. На score-панели identity-score равен ровно нулю;
логарифмическая ось не содержит ноль, поэтому tick рисуется на её левой границе
и подписывается `0 (shown at log floor)`.

Для Hydra-sweep по общим optimizer-параметрам можно дополнительно запустить:

```bash
python train_denoiser.py -m experiment=denoiser_sweep
```

Каждый Hydra job всё равно обучает полную архитектурную сетку на общих батчах.
