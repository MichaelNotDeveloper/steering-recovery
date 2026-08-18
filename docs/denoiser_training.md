# Обучение denoiser

## Что предсказывает модель

Для чистой нормализованной активации `x` online-corruptor строит

```text
c = sigma * epsilon + m * alpha * v
x_noisy = x + c
```

где `epsilon` — Gaussian noise, `v` — RMS-нормированное steering-направление,
`m` — Bernoulli-маска. Denoiser получает `x_noisy` и фактический RMS corruption
`sqrt(mean(c²))`, после чего предсказывает именно `c`. Функция потерь:

```text
L = MSE(denoiser(x_noisy, rms(c)), c)
x_recovered = x_noisy - denoiser(...)
```

Такой target делает нулевую инициализацию последнего слоя корректным стартом:
до обучения модель оставляет активацию без изменений. `identity_probability`
добавляет чистые примеры и снижает риск избыточной коррекции.

Архитектура — residual MLP с LayerNorm, gated MLP-блоками и синусоидальным
embedding логарифма noise level. Она обрабатывает token независимо и поэтому
применима как к отдельной активации, так и к `[batch, seq, hidden]`.

## Подготовка

1. Соберите hidden states ровно того слоя и модели, где будет steering.
2. Используйте `statistics.pt` от кеширования либо разрешите train script
   пересчитать статистики только по train split.
3. Сложите релевантные steering vectors в один tensor `[n_vectors, hidden_size]`.
   Если vectors не заданы, модель обучается только на Gaussian noise.
4. Проверьте, что `model.hidden_size` (если указан) совпадает с данными.

Минимальный запуск:

```bash
python train_denoiser.py \
  data.path=/data/llama-layer15 \
  data.statistics_path=/data/llama-layer15/statistics.pt \
  corruption.steering_vectors_path=/data/steering_vectors.pt \
  model.width=1024 model.depth=4 \
  training.batch_size=512 training.learning_rate=1e-4
```

Для локального smoke run:

```bash
python train_denoiser.py \
  data.path=/data/small \
  model.width=64 model.depth=2 \
  training.max_steps=20 training.batch_size=16 \
  training.precision=fp32 device=cpu wandb.mode=offline
```

## W&B

Train loop отправляет:

- `train/loss` — MSE предсказанной corruption;
- `train/noisy_mse` и `train/denoised_mse`;
- `train/relative_mse_improvement` — `1 − denoised_mse / noisy_mse`;
- `train/cosine_similarity` восстановленной и чистой активаций;
- learning rate, gradient norm и долю steering-примеров.

Validation использует отдельный детерминированный generator и те же метрики с
префиксом `val/`. Для sweep в W&B главным показателем удобно выбрать
`val/denoised_mse` (минимум), а `val/relative_mse_improvement` использовать как
диагностику. Отрицательное improvement означает, что denoiser портит активации.

Режимы W&B:

```bash
# сервер с сетью
python train_denoiser.py wandb.enabled=true wandb.mode=online

# compute node без исходящего соединения
python train_denoiser.py wandb.enabled=true wandb.mode=offline

# без W&B
python train_denoiser.py wandb.enabled=false
```

## Hydra sweep

Готовая сетка:

```bash
python train_denoiser.py -m experiment=denoiser_sweep \
  data.path=/data/llama-layer15 \
  data.statistics_path=/data/llama-layer15/statistics.pt \
  corruption.steering_vectors_path=/data/steering_vectors.pt
```

Она перебирает width/depth, learning rate, верхнюю Gaussian sigma и вероятность
steering corruption. Для короткого первого этапа ограничьте budget:

```bash
python train_denoiser.py -m experiment=denoiser_sweep \
  training.max_steps=2000 training.validation_batches=100
```

Собственная сетка без отдельного YAML:

```bash
python train_denoiser.py -m \
  model.depth=2,4,6 \
  training.learning_rate=5e-5,1e-4 \
  corruption.steering_scale_max=1,2,4
```

## Чекпоинты и продолжение анализа

- `best.pt` — минимальный `val/loss`;
- `last.pt` — последняя модель плюс optimizer/scheduler state;
- `step_N.pt` — периодические снимки;
- `statistics.pt` и `config.yaml` — полный preprocessing и параметры.

Baseline загружает `best.pt` через `denoiser.checkpoint`. Внутри hook raw steering
delta переводится в нормализованные координаты, по ней вычисляется noise level,
затем результат возвращается в исходный dtype/device LLM.

Перед большим sweep рекомендуется проверить три инварианта на небольшом наборе:

1. `val/denoised_mse < val/noisy_mse`;
2. чистые примеры при `identity_probability > 0` почти не меняются;
3. baseline при `steering.scale=0` совпадает с запуском без hook при одинаковом seed.

