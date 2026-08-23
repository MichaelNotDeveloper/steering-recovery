# Четыре метода восстановления steering

## Обозначения

Пусть `h` — hidden state до вмешательства, `v` — steering vector, `alpha` —
полная сила steering, а `D_sigma` — denoiser, обученный с уровнем шума
`sigma`. Denoiser работает с feature-wise нормализованной активацией

```text
x = (h - mean) / std.
```

В этих координатах его выход интерпретируется по формуле Tweedie:

```text
D_sigma(x) = x + sigma² * nabla log p_sigma(x).
```

Поэтому score displacement можно получить без явного деления на `sigma²`:

```text
s_sigma(x) = D_sigma(x) - x = sigma² * nabla log p_sigma(x).
```

Общий положительный множитель `sigma²` не меняет ортогональную проекцию.
Steering vector переводится в те же координаты как `v_x = v / std`.

## 1. Обычный denoising после steering

Сначала выполняется полный steering, затем denoiser:

```text
x_steered = x + alpha * v_x
x_out = D_sigma(x_steered)
```

Метод в конфиге называется `denoise`, использует `denoising_mode=full` и
`beta=1`. Denoiser может изменить hidden state в любом направлении, в том числе
частично удалить компоненту steering.

## 2. Denoising со score, ортогональным steering

После полного steering извлекается score displacement:

```text
x_steered = x + alpha * v_x
s = D_sigma(x_steered) - x_steered
s_parallel = dot(s, v_x) / dot(v_x, v_x) * v_x
s_orthogonal = s - s_parallel
x_out = x_steered + s_orthogonal
```

Метод называется `orthogonal_denoise`, использует
`denoising_mode=orthogonal` и `beta=1`. По построению denoiser не двигает
активацию вдоль steering-направления в нормализованном пространстве, но
сохраняет всю ортогональную score-компоненту.

## 3. Итеративный steering и обычный denoising

`beta` — положительное целое число подшагов. Полный steering делится на равные
части `alpha / beta`; после каждой части запускается denoiser:

```text
x_0 = x
for k = 1 .. beta:
    z_k = x_(k-1) + (alpha / beta) * v_x
    x_k = D_sigma(z_k)
x_out = x_beta
```

Метод называется `iterative_denoise` и использует `denoising_mode=full`.
Сумма добавленных steering-сдвигов до поправок denoiser остаётся равной
`alpha * v_x`.

## 4. Итеративный steering и ортогональный score

Steering также делится на `beta` равных частей, но на каждом подшаге из score
удаляется параллельная steering-компонента:

```text
x_0 = x
for k = 1 .. beta:
    z_k = x_(k-1) + (alpha / beta) * v_x
    s_k = D_sigma(z_k) - z_k
    s_k_orthogonal = s_k - dot(s_k, v_x) / dot(v_x, v_x) * v_x
    x_k = z_k + s_k_orthogonal
x_out = x_beta
```

Метод называется `iterative_orthogonal_denoise` и использует
`denoising_mode=orthogonal`. Это наиболее строгий вариант: каждый из `beta`
вызовов denoiser сохраняет steering-компоненту текущего подшага.

## Benchmark-сетка

`configs/steering_benchmark.yaml` строит декартово произведение:

```text
6 denoisers x 4 recovery algorithms x 4 steering vectors x 5 alpha values
```

Шесть checkpoint — это `latent_dim=3072`, три residual block и уровни шума
`sigma = 0.1, 0.2, 0.5`: три модели без dropout из `runs/denoiser` и три с
`dropout=0.1` из `runs/drouput_run`. Базовый `raw` steering запускается
дополнительно. Один checkpoint загружается один раз, после чего для него
последовательно выполняются все четыре алгоритма. Во время benchmark все
модели находятся в `eval`, поэтому dropout-модели дают детерминированный
denoiser output; dropout здесь является способом обучения, а не MC-sampling.

Значение `beta` общее для двух итеративных методов:

```bash
python run_steering_benchmarks.py recovery.beta=8
```

Пути к двум группам моделей вынесены в
`recovery.standard_run_dir` и `recovery.dropout_run_dir`, поэтому при переносе
checkpoint не требуется менять 24 benchmark-метода.

Прямые методы всегда используют `beta=1`. Для одного intervention итеративный
метод делает ровно `beta` вызовов denoiser, что сохраняется в поле
`denoiser_calls` condition JSONL и отображается в HTML.

## Имена и отчёты

Имя benchmark-метода составляется как
`<denoiser_name>__<algorithm_name>`, например
`dropout_sigma_0p2__iterative_orthogonal_denoise`. Благодаря этому существующий
plotter автоматически строит для каждого из четырёх вариантов те же графики
target probability против Dist-1, Dist-2, Dist-3 и SLOR, что и для обычного
steering.

`examples.html` содержит отдельные фильтры и пометки для denoiser, режима
восстановления, `beta`, `sigma` и dropout. Полные параметры также сохраняются в
`summary.jsonl`, `summary.csv`, condition JSONL и `manifest.json`.
