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

Обучение denoiser напрямую на streaming OpenWebText:

```bash
python train_denoiser.py \
  data.streaming.model_name=gpt2 \
  data.streaming.layer_index=6 \
  training.batch_size=512 \
  training.max_steps=10000
```

`training.batch_size` — точное число hidden states, которое `IterableDataset`
выдаёт за одну итерацию. Неполный остаток переносится между текстами.

Опциональный статический режим:

```bash
python train_denoiser.py \
  data.mode=static \
  data.path=/path/to/activations \
  data.statistics_path=/path/to/activations/statistics.pt \
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
