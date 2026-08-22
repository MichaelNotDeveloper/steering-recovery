# Логистические классификаторы тем AG News

## Назначение

`train_topic_logistic_regressions.py` независимо от Difference-of-Means запуска
обучает четыре one-vs-rest L2 logistic regression для тем `World`, `Sports`,
`Business` и `Sci/Tech`. Признаки — FP32 hidden-состояния всех настоящих токенов
полного `description` на выходе `h[5]` GPT-2 Small. Padding не входит в выборку.

## Выборка и обучение

Сначала GPT-2 обрабатывает одинаковое число статей каждого класса. Для каждого
класса random-priority reservoir равномерно оставляет ровно
`sampling.hidden_states_per_class` token hidden states. Поэтому объём train
sample строго одинаков для всех четырёх классов и не зависит от средней длины
статей.

Каждый optimization batch также сбалансирован: он содержит
`training.batch_size_per_class` примеров каждого класса, то есть эффективный
batch size в четыре раза больше. Четыре независимых выхода оптимизируют

```text
BCEWithLogits + λ/2 · ||w||²
```

Bias не регуляризуется. После каждой эпохи на полной сбалансированной train
выборке считаются loss, ROC-AUC и trapezoidal AUC-PRC отдельно для каждой темы и
macro-average. Validation split намеренно не используется; это явно записано в
metadata и checkpoint.

## Запуск

Полный запуск с настройками по умолчанию:

```bash
python train_topic_logistic_regressions.py
```

По умолчанию обрабатываются 10 000 статей каждого класса, из каждого класса
сохраняется reservoir на 100 000 hidden-состояний, затем выполняются 10 эпох.
Пример более короткого smoke-прогона:

```bash
python train_topic_logistic_regressions.py \
  sampling.articles_per_class=32 \
  sampling.hidden_states_per_class=512 \
  training.epochs=2 \
  training.batch_size_per_class=64 \
  output_dir=data/topic_logistic_regression/ag_news/smoke
```

GPT-2 Small принудительно работает в `float32`: override с `fp16`/`bf16`
завершится ошибкой до загрузки модели.

## Артефакты

Результаты по умолчанию находятся в
`data/topic_logistic_regression/ag_news/gpt2_layer_5/`:

```text
config.yaml
logistic_business.pt
logistic_regressions.pt
logistic_sci_tech.pt
logistic_sports.pt
logistic_world.pt
manifest.json
token_probability_examples.html
training_curves.png
training_history.json
```

`logistic_regressions.pt` содержит `weights: Tensor[4, 768]`, `bias: Tensor[4]`,
порядок тем, sampling metadata и метрики всех эпох. Матрица весов также доступна
под ключом `steering_vectors`, а отдельные тематические файлы — под ключом
`steering_vector`, поэтому артефакты совместимы с существующим benchmark loader.

`training_history.json` хранит loss, `*_roc_auc`, `*_auc_prc`,
`macro_roc_auc` и `macro_auc_prc` для каждой эпохи. `training_curves.png`
показывает три панели: loss, ROC-AUC и AUC-PRC.

## HTML токенных вероятностей

`token_probability_examples.html` — самодостаточный файл без внешних assets. В
него входят четыре случайных примера каждого истинного класса из test split.
Переключатель «Подсветка классификатора» даёт четыре режима — по одному для
вероятности `World`, `Sports`, `Business` и `Sci/Tech`. Второй переключатель
фильтрует карточки по истинному классу.

Для каждого токена сохранены позиция, GPT-2 token ID, декодированный текст и все
четыре вероятности; они видны в tooltip. В metadata каждой карточки находятся
исходный текст, split, истинная метка, число токенов, модель, слой, dtype и имя
checkpoint. Полные параметры запуска доступны в верхнем блоке metadata.
