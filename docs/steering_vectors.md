# Генерация steering-векторов

Код генерации направлений находится в `steering_recovery/steering/`. Папка
отделена от обучения denoiser и baseline-генерации; здесь же будут размещаться
бенчмарки методов steering. Общая часть пайплайна не зависит от AG News:

1. адаптер датасета выдаёт пары `(label, text)`;
2. сборщик набирает точную квоту статей для каждой группы;
3. token builder передаёт полный текст в пределах контекста модели;
4. extractor снимает hidden каждого настоящего токена выбранного block;
5. произвольные positive/negative группы задаются через `ContrastDefinition`.

## AG News: темы one-vs-rest

Конфигурация по умолчанию использует
[`sh0416/ag_news`](https://huggingface.co/datasets/sh0416/ag_news), split `train`
и поле `description`. Поле `title` не добавляется: в этой конфигурации
`description` считается текстом статьи. Датасет перемешивается с фиксированным
`seed=42`, после чего собирается по 1000 примеров каждой исходной метки:

| Label | Тема | Файл |
|---:|---|---|
| 1 | World | `world.pt` |
| 2 | Sports | `sports.pt` |
| 3 | Business | `business.pt` |
| 4 | Sci/Tech | `sci_tech.pt` |

По умолчанию используется `extraction.mode=full_text_all_tokens`. Поле
`description` токенизируется целиком без дополнительного prompt, а GPT-2
получает все token IDs вплоть до контекстного предела `1024`. Из выхода `h[5]`
берутся все позиции настоящих токенов; padding исключается.

Difference of Means является token-weighted. Если тема `c` содержит набор
token hidden states `H_c`, направление вычисляется как

```text
v_c = mean(H_c) - mean(H_not_c)
```

Квота `collection.samples_per_topic=1000` относится к числу статей. Реальное
число hidden states равно числу обработанных токенов и сохраняется отдельно для
каждой темы. Моменты считаются потоково в `float64` формулами Chan/Welford.

Предыдущий способ остаётся доступен через
`extraction.mode=prompt_last_token`. В нём
берутся первые 48 GPT-2 token IDs и строится prompt:

```text
Article: {первые 48 GPT-2 токенов description}
This article is mainly about
```

Forward hook установлен на `h[5]`, то есть на шестой transformer block GPT-2.
Из его выхода берётся позиция последнего настоящего токена prompt — токена
` about`. При запуске это проверяется по tokenizer; padding не участвует в
выборе.

GPT-2 Small загружается в `float32`; dtype записывается в metadata каждого
артефакта. Векторы из `fp16`/`bf16` hidden states несовместимы с FP32-бенчмарком.

## Запуск

```bash
python generate_steering_vectors.py
```

Для smoke-прогона с малыми квотами и отдельной директорией:

```bash
python generate_steering_vectors.py \
  collection.samples_per_topic=8 \
  collection.batch_size=8 \
  output_dir=data/steering_vectors/ag_news/smoke
```

Параметры модели, слоя, extraction, колонок датасета и квот переопределяются
обычными Hydra overrides. Например, чтобы использовать заголовок вместо описания:

```bash
python generate_steering_vectors.py dataset.text_column=title
```

## Артефакты

По умолчанию файлы записываются в
`data/steering_vectors/ag_news/gpt2_layer_5/`:

```text
business.pt
config.yaml
manifest.json
sci_tech.pt
sports.pt
steering_vectors.pt
world.pt
```

Канонические FP32-векторы, рассчитанные по 10 000 статей каждого класса, включены
в Git вместе с `manifest.json` и фактически использованным `config.yaml`.
Повторный запуск не нужен для benchmark; команда выше нужна для пересчёта или
новой конфигурации.

Каждый тематический `.pt` содержит ключ `steering_vector: Tensor[768]` и поэтому
непосредственно совместим с `baseline.steering.vector.key=steering_vector`.
Рядом сохранены positive/negative means, размеры выборок и полные метаданные:
датасет, seed, модель, tokenizer, индекс слоя, режим извлечения и метод расчёта.

`steering_vectors.pt` содержит общую матрицу `[4, 768]` под ключом
`steering_vectors`, имена/метки направлений, групповые mean/variance/count и те
же метаданные. `manifest.json` — человекочитаемое описание файлов и L2-норм
векторов, а `config.yaml` — фактически использованный Hydra-конфиг.

Логистические направления больше не обучаются в этом запуске. Для них есть
отдельный epoch-based pipeline, описанный в
[документации классификаторов](topic_logistic_regression.md).

Чтобы использовать, например, World-вектор в существующем baseline GPT-2,
нужно согласовать модель и слой вмешательства:

```bash
python run_baselines.py \
  model.name=gpt2 \
  steering.vector.path=data/steering_vectors/ag_news/gpt2_layer_5/world.pt \
  steering.layer_path=transformer.h \
  steering.layer_index=5 \
  steering.scale=1
```

## Добавление нового источника

Новый метод поиска не должен дублировать batching и hook. Достаточно добавить
адаптер, который выдаёт `LabeledText`, определить группы и contrasts, затем
передать их в `collect_group_token_moments` и `compute_contrasts`. Если у задачи не
one-vs-rest разметка, `ContrastDefinition` позволяет указать несколько
positive и negative labels явно. Новый generator регистрируется в
`steering_recovery/steering/pipeline.py`.
