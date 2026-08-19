# Steering Recovery

Воспроизводимый пайплайн для экспериментов с activation steering:

- streaming teacher-forced hidden states из OpenWebText через GPT-2;
- опциональное кеширование hidden states в `.npy`;
- baseline-генерация без вмешательства и с режимами `once_at_start`,
  `every_step`, `entropy_threshold`;
- обучение residual denoiser на Gaussian- и steering-коррупциях;
- Hydra-конфигурации и multirun-перебор гиперпараметров;
- метрики и артефакты в Weights & Biases.

## Быстрый старт

```bash
conda env create -f environment.yaml
conda activate steering-recovery
```

Сначала соберите статистики GPT-2 hidden states. Лимит считается по реально
добавленным hidden-токенам (padding и первый токен каждого текста не входят):

```bash
python collect_hidden_statistics.py \
  source.layer_index=6 \
  collection.max_tokens=1000000 \
  output_path=data/gpt2_layer_6_statistics.pt
```

Сбор идёт с `tqdm`; промежуточные моменты считаются в `float64` алгоритмом
Chan/Welford без хранения активаций. Затем запустите обучение:

```bash
python train_denoiser.py \
  data.streaming.model_name=gpt2 \
  data.streaming.layer_index=6 \
  data.statistics_path=data/gpt2_layer_6_statistics.pt \
  training.batch_size=512 \
  training.max_steps=10000
```

`training.batch_size` — точное число hidden states, которое `IterableDataset`
выдаёт за одну итерацию. Неполный остаток переносится между текстами.
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
  data.statistics_path=/path/to/gpt2_layer_6_statistics.pt \
  corruption.steering_vectors_path=/path/to/vectors.pt
```

Baseline без steering:

```bash
python run_baselines.py \
  data.prompts_path=/path/to/prompts.jsonl \
  steering.mode=once_at_start steering.scale=0 \
  wandb.enabled=false
```

Все параметры можно переопределять из CLI. Подробности: [архитектура](docs/architecture.md)
и [обучение denoiser](docs/denoiser_training.md).

## Проверки

```bash
pytest
```
