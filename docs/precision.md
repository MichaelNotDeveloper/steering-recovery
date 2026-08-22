# Политика точности GPT-2 Small

Все эксперименты, в которых используется `gpt2` (GPT-2 Small), выполняются в
`float32`. Это относится к весам модели, hidden states, logits, steering-векторам,
обучению/валидации denoiser и итоговой генерации. Сниженная точность меняет
предсказания GPT-2 и поэтому не считается совместимой с FP32-результатами.

## Защита от смешивания результатов

- В основных Hydra-конфигах явно задан `float32`; для denoiser также задано
  `training.precision=fp32`.
- Общие загрузчики распознают `gpt2` и `openai-community/gpt2`: `auto`
  разрешается как FP32, а явные `fp16`/`bf16` завершают запуск с ошибкой.
- Статистики hidden states сохраняют фактический `model_dtype`. Обучение
  denoiser проверяет его вместе с моделью, слоем и длиной контекста.
- Steering benchmark отклоняет векторы, полученные не в FP32, и checkpoints
  denoiser, обученные не полностью в FP32.
- Frozen AG News classifier в benchmark также запускается в FP32, чтобы
  вероятности классов были воспроизводимы.
- GPT-2 Large, используемый только для SLOR в benchmark, запускается в FP16;
  его log-softmax reduction считается в FP32, а веса не смешиваются в памяти с
  генеративным GPT-2 Small.

## Какие артефакты нужно пересчитать

Изменение precision инвалидирует всю зависимую цепочку:

1. `data/gpt2_layer_5_statistics.pt` — собрать заново через
   `collect_hidden_statistics.py`;
2. grid denoiser — переобучить через `train_denoiser.py` на новых статистиках;
3. AG News steering-векторы — пересчитать через
   `generate_steering_vectors.py`;
4. steering benchmark — перезапустить через `run_steering_benchmarks.py` с
   FP32-векторами и новыми FP32-checkpoints denoiser.

Старые summary, plots и checkpoints нельзя сравнивать с новым прогоном как
результаты одного precision-режима.

Из `data/` в Git хранится только канонический
`data/gpt2_layer_5_statistics.pt`. Steering-векторы, `runs/` и `comparison/`
остаются локальными: после клонирования их нужно получить FP32-пайплайном по
цепочке выше.
