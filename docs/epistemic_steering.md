# MC-dropout epistemic steering

## Цель эксперимента

Эксперимент измеряет, насколько необычными для denoiser становятся hidden states
после сильного steering. Он использует четыре FP32 Difference-of-Means вектора
AG News, `alpha=[6,8,10,20]` и три denoiser для `sigma=[0.1,0.2,0.5]`.

Каждый residual block denoiser имеет вид:

```text
Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768) -> Dropout(0.1) + skip
```

Модель содержит три таких блока. Dropout работает во время train и принудительно
включается только внутри MC-inference; GPT-2 и все вычисления остаются в
`float32`.

## Этап 1: обучение denoiser

```bash
python train_denoiser.py experiment=epistemic_dropout
```

Preset `configs/experiment/epistemic_dropout.yaml` обучает ровно три модели:

```text
latent_3072_layers_3_sigma_0p1_dropout_0p1
latent_3072_layers_3_sigma_0p2_dropout_0p1
latent_3072_layers_3_sigma_0p5_dropout_0p1
```

Используются обычные training/validation настройки из `configs/denoiser.yaml`.
При необходимости их можно переопределять в той же команде.

## Этап 2: steering и MC-dropout

Передайте директорию первого запуска:

```bash
python run_epistemic_steering.py \
  denoiser_run_dir=runs/epistemic-denoisers/YYYY-MM-DD/HH-MM-SS
```

Runner проверяет архитектуру, dropout, `sigma`, source model/layer, FP32
provenance и SHA256 каждого checkpoint. Несовместимый checkpoint отклоняется до
генерации.

Для каждого сочетания из трёх denoiser, четырёх векторов и четырёх alpha
используются одни и те же 100 стратифицированных AG News prompts: первые 24
GPT-2 токена статьи. Генерируются 40 новых токенов. На каждом шаге:

1. из `h[5]` берётся hidden последней позиции и применяется `h + alpha * v`;
2. hidden нормализуется сохранёнными train-статистиками denoiser;
3. выполняются 20 forward с разными dropout masks;
4. среднее из 20 предсказаний денормализуется и передаётся следующим слоям
   GPT-2;
5. четыре статистики прикрепляются к предсказанному на этом шаге токену.

Полный запуск содержит 48 условий, 4 800 генераций, 192 000 диагностированных
token hidden states и 3 840 000 MC-предсказаний denoiser. Условия сохраняются
раздельно и поддерживают resume. Быстрый smoke-вариант:

```bash
python run_epistemic_steering.py \
  denoiser_run_dir=/path/to/training/run \
  generation.samples_per_condition=4 \
  generation.new_tokens=4 \
  mc_dropout.samples=3 \
  examples.per_condition=1
```

## Статистики

Пусть `z = normalize(h + alpha * v)`, а `D_i(z)` — результат `i`-го из `M=20`
dropout forward. По формуле Tweedie в нормализованных координатах:

```text
delta_i = D_i(z) - z = sigma² * nabla log p_sigma(z)
```

Сохраняются:

```text
score_mean_deviation = mean_i ||delta_i - mean(delta)||_2
score_length_variance = Var_i[||delta_i||_2]
cosine_i = 1 - cosine(delta_i, mean(delta))
score_cosine_distance_variance = Var_i[cosine_i]
prediction_mean_deviation = mean_i ||D_i(z) - mean(D(z))||_2
```

Все variance являются population variance (`correction=0`). Первые три метрики
характеризуют неопределённость `sigma² * score`, четвёртая — полный разброс
предсказаний denoiser. В raw GPT-координаты метрики не переводятся, чтобы их
масштаб соответствовал `sigma`, на котором обучалась модель.

## Результаты

Hydra сохраняет:

```text
run/
├── config.yaml
├── prompts.jsonl
├── prompts_metadata.json
├── conditions/sigma_<sigma>/<vector>/alpha_<alpha>.jsonl
├── condition_summary.json
├── condition_summary.csv
├── plots/
│   ├── score_mean_deviation.png
│   ├── score_length_variance.png
│   ├── score_cosine_distance_variance.png
│   └── prediction_mean_deviation.png
├── examples.html
└── manifest.json
```

Каждый график содержит три панели по `sigma`; линии соответствуют четырём
steering-векторам, ось X — `alpha`, полупрозрачная область — межквартильный
диапазон токенных значений. Общий диапазон оси Y вычисляется по mean и границам
IQR сразу для всех `sigma` с дополнительным отступом, поэтому линии и области
не обрезаются и остаются сравнимыми между панелями.

`examples.html` содержит четыре примера на каждое условие — по одному исходному
классу AG News. Можно выбрать `sigma`, вектор, `alpha` и одну из четырёх
статистик. Подсветка токена нормируется по общему 5–95% диапазону выбранной
метрики; tooltip показывает точные значения всех статистик. В карточке также
доступны prompt, seeds, checkpoint, SHA256 и полные метаданные генерации.
