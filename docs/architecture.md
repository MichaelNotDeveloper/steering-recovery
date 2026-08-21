# Архитектура проекта и работа с пайплайном

## Назначение

Проект разделяет streaming-получение данных, обучение denoiser и генеративные
эксперименты. GPT-2 работает в `eval`/`inference_mode`, не обучается и производит
teacher-forced hidden states непосредственно во время обучения. Denoiser —
единственный trainable-модуль и позднее подключается к baseline forward hook.

## Структура

```text
steering-recovery/
├── cache_activations.py       # Hydra-entrypoint сбора hidden states
├── collect_hidden_statistics.py # streaming sum/variance GPT-2 hidden states
├── compare_denoisers.py       # CSV/Markdown/barplot по model folders
├── generate_steering_vectors.py # Hydra-entrypoint поиска steering-векторов
├── train_denoiser.py          # Hydra-entrypoint обучения
├── run_baselines.py           # Hydra-entrypoint генерации
├── configs/
│   ├── cache_activations.yaml
│   ├── denoiser.yaml
│   ├── baseline.yaml
│   └── experiment/            # готовые параметры multirun
├── steering_recovery/
│   ├── cache.py               # hook и запись .npy shards
│   ├── data.py                # lazy dataset и статистики
│   ├── streaming_data.py      # IterableDataset: OpenWebText → GPT-2 → batches
│   ├── statistics.py          # Chan/Welford и загрузка нормализации
│   ├── denoiser.py            # residual MLP
│   ├── training.py            # train/validation/checkpoints
│   ├── comparison.py          # агрегация сохранённых model summaries
│   ├── intervention.py        # политики steering
│   ├── generation.py          # autoregressive loop с KV-cache
│   ├── baseline.py            # оркестрация baseline
│   ├── steering/              # генерация векторов и будущие бенчмарки
│   │   ├── core.py            # prompt, hook, квоты групп и contrasts
│   │   ├── ag_news.py         # адаптер и one-vs-rest темы AG News
│   │   ├── artifacts.py       # .pt-векторы и manifest с метаданными
│   │   └── pipeline.py        # registry/dispatch генераторов
│   └── reporting.py           # JSONL и HTML-отчёт
├── docs/
└── tests/
```

Поток данных:

```mermaid
flowchart LR
    A["Streaming OpenWebText"] --> B["GPT-2 teacher-forced forward"]
    B --> C["IterableDataset: exact k hidden states"]
    C --> D["Normalize once"]
    D --> E["Shared epsilon; sigma 0.1 / 0.2 / 0.5"]
    E --> J["27 residual MLP models in lockstep"]
    B -. "optional" .-> I["Activation shards"]
    F["Prompts + steering vector"] --> G["Baseline generation"]
    J -->|"best model checkpoint"| G
    G --> H["JSONL + HTML + W&B"]
    K["AG News labeled articles"] --> L["GPT-2 h[5] hidden over 'about'"]
    L --> M["Four one-vs-rest mean differences"]
    M --> N["data/steering_vectors + metadata"]
```

## Форматы данных

### Streaming teacher-forced dataset

Основной режим — `data.mode=streaming`. Для каждого набора из
`data.streaming.text_batch_size` документов выполняется один causal forward
GPT-2 по настоящей последовательности token IDs. Генерация новых token не
используется: hidden в позиции `t` вычисляется по настоящим токенам вплоть до
позиции `t` включительно.

Из каждого текста берутся все непаддинговые hidden states, кроме самого первого
реального token. Это делается по `attention_mask`, поэтому padding не влияет на
выбор. Однотокенные и пустые тексты не добавляют активаций.

`TeacherForcedActivationIterableDataset` самостоятельно формирует batches:

1. получает очередные тексты из `Skylion007/openwebtext` через
   `load_dataset(..., streaming=True)`;
2. извлекает активации выбранного GPT-2 block;
3. последовательно складывает token states в буфер;
4. выдаёт tensor строго `[k, hidden_size]`, где
   `k = training.batch_size`;
5. переносит остаток к следующему документу; только последний неполный остаток
   всего stream отбрасывается.

Dataset уже выполняет batching, поэтому train loop итерирует его напрямую.
При ручном использовании `DataLoader` допустимы только
`batch_size=None, num_workers=0`: iterator владеет GPT-2 model и не должен
копироваться в worker processes.

Первые `data.streaming.validation_texts` документов выделяются в стабильный
validation stream. Training stream начинается после них и перемешивается
ограниченным буфером `shuffle_buffer_size`; seed меняется с эпохой.

GPT-2 имеет предел контекста 1024. Тексты обрезаются до
`data.streaming.max_length`, и внутри получившейся последовательности
загружаются все states кроме первого.

### Статистики hidden states

До обучения запускается `collect_hidden_statistics.py`. Он использует тот же
teacher-forced extractor и останавливается ровно после
`collection.max_tokens` hidden-токенов. `tqdm` показывает прогресс в этих
токенах, а не в документах.

Для каждого feature хранятся `sum: float64[hidden_size]` и
`variance: float64[hidden_size]`, а также скалярный `count`. Внутреннее состояние
сборщика — `(count, mean, M2)`; батчи объединяются формулами Chan/Welford в
`float64`. Поэтому код не использует нестабильную формулу
`E[x²] - E[x]²`, не накапливает большой raw sum на каждом шаге и требует
`O(hidden_size)` постоянной памяти помимо текущего batch.

При загрузке denoiser получает `mean = sum / count` и
`std = sqrt(variance)`. Отсутствующий файл, один из ключей, несовпадающий
`hidden_size` или другая source model/layer считаются ошибкой — online-пересчёта
во время обучения нет.

### Grid residual MLP

После общей нормализации один batch используется всеми моделями. Для каждого
шага создаётся один `epsilon ~ N(0, I)`; варианты получают
`x_noisy = x + sigma * epsilon` и учатся напрямую восстанавливать `x` по MSE.

Residual block состоит только из
`Linear(768, latent_dim, bias=True) → GELU → Linear(latent_dim, 768, bias=True)`
и skip connection. Внутреннего LayerNorm нет. Grid содержит 27 моделей:
`latent_dim=[192,768,3072]`, `num_layers=[1,3,5]`,
`sigma=[0.1,0.2,0.5]`.

Validation выполняется каждые `training.validation_every_batches` батчей.
Каждая модель имеет отдельную папку с `metrics.jsonl`, `summary.json`, лучшим
`best.pt` по validation L2 и финальным `last.pt`.

### Статические активации

Режим `data.mode=static` оставлен для воспроизводимых offline-прогонов.
`data.path` принимает один `.npy`, `.pt`, `.pth` или директорию с shards.
Каждый tensor должен иметь форму `[..., hidden_size]`; все ведущие измерения
считаются независимыми примерами. Для `.pt` допустим tensor либо словарь с ключом
из `data.key` (по умолчанию `activations`). Обычные `.npy` читаются через memory
map.

Опциональный `cache_activations.py` создаёт:

- `activations_00000.npy`, ... — shards;
- `statistics.pt` — устойчиво рассчитанные `sum`, `variance` и `count`;
- `manifest.json` — размерность, число примеров, список shards и полный config;
- `config.yaml` — фактически использованная конфигурация.

### Steering vectors

Один vector: tensor `[hidden_size]` или словарь с `steering_vector`.
Набор направлений для обучения: `[n_vectors, hidden_size]` или словарь с
`steering_vectors`. Слой и hidden size должны совпадать с кешированными
активациями.

Генераторы и будущие бенчмарки steering изолированы в
`steering_recovery/steering/`. Первый генератор строит четыре AG News
направления как `mean(topic) - mean(other topics)` по hidden последнего токена
prompt на выходе `h[5]` GPT-2. Детальный формат артефактов и запуск описаны в
[отдельном документе](steering_vectors.md).

### Prompts

Рекомендуемый JSONL:

```json
{"id":"task-001","prompt":"Полный prompt модели"}
```

Остальные поля сохраняются во входном наборе, но baseline использует `id` и
`prompt`. Также поддерживаются JSON-массив и текстовый файл (один prompt в строке).

## Запуск на сервере

Установка:

```bash
conda env create -f environment.yaml
conda activate steering-recovery
```

Для закрытых моделей заранее выполните `huggingface-cli login`. W&B использует
обычный `wandb login`; без сети задайте `wandb.mode=offline`, а для полного
отключения — `wandb.enabled=false`.

Сбор статистик GPT-2 `layer_index=5` (6-й блок):

```bash
python collect_hidden_statistics.py \
  source.model_name=gpt2 \
  source.layer_path=h \
  source.layer_index=5 \
  collection.max_tokens=1000000 \
  collection.batch_tokens=4096 \
  output_path=data/gpt2_layer_5_statistics.pt
```

Streaming-обучение на том же слое:

```bash
python train_denoiser.py \
  data.mode=streaming \
  data.streaming.dataset_name=Skylion007/openwebtext \
  data.streaming.model_name=gpt2 \
  data.streaming.layer_path=h \
  data.streaming.layer_index=5 \
  data.statistics_path=data/gpt2_layer_5_statistics.pt \
  data.streaming.max_length=1024 \
  data.streaming.text_batch_size=8 \
  training.batch_size=512 \
  training.max_steps=10000
```

`AutoModel(gpt2)` предоставляет blocks как `h[0] ... h[11]`; поэтому source
dataset использует `layer_path=h`. В causal LM baseline те же blocks находятся
по пути `transformer.h`.

Модель, слой и `max_length` сборщика должны совпадать с параметрами training
stream. Файл статистик обязателен и копируется в директорию конкретного run.

Статический запуск на ранее сохранённых активациях:

```bash
python train_denoiser.py \
  data.mode=static \
  data.path=/data/activations \
  data.statistics_path=/data/activations/statistics.pt
```

Baseline без вмешательства и с вмешательством:

```bash
python run_baselines.py \
  model.name=/models/llama \
  data.prompts_path=/datasets/prompts.jsonl \
  steering.scale=0 wandb.enabled=false

python run_baselines.py \
  model.name=/models/llama \
  data.prompts_path=/datasets/prompts.jsonl \
  steering.vector.path=/vectors/target.pt \
  steering.layer_index=15 \
  steering.mode=once_at_start steering.scale=1
```

Подключение denoiser в том же прогоне:

```bash
python run_baselines.py \
  model.name=/models/llama \
  data.prompts_path=/datasets/prompts.jsonl \
  steering.vector.path=/vectors/target.pt steering.scale=1 \
  denoiser.enabled=true \
  denoiser.checkpoint=/runs/denoiser/models/latent_192_layers_1_sigma_0p1/best.pt
```

Сводная таблица и barplot по всем model folders:

```bash
python compare_denoisers.py /runs/denoiser/<run> --output-dir comparison
```

Режимы вмешательства:

- `none` — hook установлен, но активации не меняются;
- `once_at_start` — vector добавляется один раз на prefill;
- `every_step` — vector добавляется на каждом forward;
- `entropy_threshold` — vector добавляется, если нормализованная token entropy
  предыдущего шага не меньше порога. Это причинная политика с KV-cache; prefill
  не изменяется, а первая возможная интервенция происходит на следующем forward.

Baseline сохраняет `generations.jsonl`, `metrics.jsonl`, `report.html` и полный
`config.yaml`. Основные агрегаты: длина, средняя нормализованная entropy и доля
forward-вызовов с интервенцией. Семантический judge намеренно не зашит в проект:
его ответы можно объединять с `generations.jsonl`, не смешивая генерацию с
конкретным внешним API.

## Воспроизводимость и Hydra

Одиночный override:

```bash
python run_baselines.py steering.scale=10 generation.seed=123
```

Grid по режиму, scale и порогу:

```bash
python run_baselines.py -m \
  steering.mode=once_at_start,entropy_threshold \
  steering.scale=0,1,10 \
  steering.entropy_threshold=0.25,0.35,0.45
```

Каждый job получает отдельную директорию Hydra. В config результата записаны все
override, seed и пути к артефактам.
