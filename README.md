# Steering Recovery

Воспроизводимый пайплайн для экспериментов с activation steering:

- кеширование hidden states выбранного слоя LLM;
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

Кеширование активаций:

```bash
python cache_activations.py \
  model.name=meta-llama/Llama-3.1-8B-Instruct \
  capture.layer_index=15 \
  dataset.path=HuggingFaceFW/fineweb \
  dataset.name=sample-10BT
```

Обучение denoiser:

```bash
python train_denoiser.py \
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

