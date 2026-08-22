from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from steering_recovery.steering.core import TopicDefinition


def write_token_probability_report(
    path: str | Path,
    *,
    topics: Sequence[TopicDefinition],
    examples: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    """Write a self-contained token-level classifier probability report."""

    if not topics:
        raise ValueError("at least one report topic is required")
    topic_payload = [
        {"label": topic.label, "name": topic.name, "slug": topic.slug}
        for topic in topics
    ]
    payload = {
        "topics": topic_payload,
        "examples": [dict(example) for example in examples],
        "metadata": dict(metadata),
    }
    serialized = (
        json.dumps(payload, ensure_ascii=False)
        .replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )
    html = _HTML_TEMPLATE.replace("__REPORT_PAYLOAD__", serialized)

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AG News · token probabilities</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #637083;
      --line: #dce3ed;
      --panel: #ffffff;
      --canvas: #f4f6fa;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--canvas); color: var(--ink); }
    main { width: min(1180px, calc(100% - 32px)); margin: 38px auto 64px; }
    h1 { margin: 0 0 8px; font-size: clamp(27px, 4vw, 42px); letter-spacing: -0.035em; }
    .lead { color: var(--muted); margin: 0; max-width: 790px; line-height: 1.55; }
    .summary { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
    .pill { padding: 7px 11px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); font-size: 13px; }
    .toolbar { position: sticky; top: 12px; z-index: 4; display: grid; grid-template-columns: 1fr 1fr auto; gap: 14px; align-items: end; margin: 24px 0; padding: 16px; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.94); box-shadow: 0 10px 30px rgba(41,55,78,.08); backdrop-filter: blur(10px); }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
    select { width: 100%; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 10px; background: white; color: var(--ink); font: inherit; }
    .legend { display: grid; grid-template-columns: auto 120px auto; gap: 7px; align-items: center; font-size: 12px; color: var(--muted); }
    .gradient { height: 10px; border-radius: 999px; background: linear-gradient(90deg, hsla(217,84%,55%,.05), hsla(217,84%,55%,.95)); }
    #cards { display: grid; gap: 15px; }
    article { padding: 20px; background: var(--panel); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 7px 23px rgba(33,48,72,.045); }
    .card-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; margin-bottom: 15px; }
    .card-head h2 { margin: 0; font-size: 18px; }
    .card-subtitle { color: var(--muted); font-size: 13px; }
    .tokens { line-height: 2.05; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .token { border-radius: 4px; padding: 2px 0; box-decoration-break: clone; -webkit-box-decoration-break: clone; transition: background-color .15s ease; }
    details { margin-top: 14px; color: var(--muted); font-size: 13px; }
    summary { cursor: pointer; font-weight: 700; color: #445167; }
    pre { max-height: 330px; overflow: auto; padding: 13px; border-radius: 10px; background: #f7f9fc; color: #334155; font-size: 12px; white-space: pre-wrap; }
    .empty { padding: 36px; text-align: center; color: var(--muted); background: white; border: 1px dashed #cbd5e1; border-radius: 15px; }
    @media (max-width: 720px) { .toolbar { position: static; grid-template-columns: 1fr; } .legend { grid-template-columns: auto 1fr auto; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Token-level AG News probabilities</h1>
    <p class="lead">Каждый токен окрашен по вероятности, которую ему назначает выбранная one-vs-rest логистическая регрессия. Наведите курсор на токен, чтобы увидеть все четыре вероятности.</p>
    <div class="summary" id="summary"></div>
    <details><summary>Метаданные запуска</summary><pre id="reportMetadata"></pre></details>
  </header>
  <section class="toolbar" aria-label="Report controls">
    <label>Подсветка классификатора<select id="highlight"></select></label>
    <label>Истинный класс<select id="trueClass"><option value="all">Все классы</option></select></label>
    <div class="legend"><span>0</span><div class="gradient" id="gradient"></div><span>1</span></div>
  </section>
  <section id="cards"></section>
</main>
<script>
const DATA = __REPORT_PAYLOAD__;
const hues = [217, 145, 31, 286];
const highlight = document.querySelector('#highlight');
const trueClass = document.querySelector('#trueClass');
const cards = document.querySelector('#cards');
const topicIndex = new Map(DATA.topics.map((topic, index) => [String(topic.label), index]));

for (const topic of DATA.topics) {
  for (const select of [highlight, trueClass]) {
    const option = document.createElement('option');
    option.value = String(topic.label);
    option.textContent = topic.name;
    select.append(option);
  }
}

document.querySelector('#reportMetadata').textContent = JSON.stringify(DATA.metadata, null, 2);
const summaryValues = [
  `${DATA.examples.length} примеров`,
  `${DATA.topics.length} классификатора`,
  `${DATA.metadata.source?.model_name ?? 'model'} · h[${DATA.metadata.source?.layer_index ?? '?'}]`,
  `${DATA.metadata.source?.model_dtype ?? 'dtype unknown'}`,
];
for (const value of summaryValues) {
  const node = document.createElement('span');
  node.className = 'pill';
  node.textContent = value;
  document.querySelector('#summary').append(node);
}

function probabilityTitle(token, probabilities) {
  const lines = [`token #${token.index} · id ${token.id}`, JSON.stringify(token.text)];
  DATA.topics.forEach((topic, index) => lines.push(`${topic.name}: ${probabilities[index].toFixed(5)}`));
  return lines.join('\n');
}

function render() {
  cards.replaceChildren();
  const selectedLabel = highlight.value || String(DATA.topics[0].label);
  const selectedIndex = topicIndex.get(selectedLabel);
  const hue = hues[selectedIndex % hues.length];
  document.querySelector('#gradient').style.background = `linear-gradient(90deg, hsla(${hue},84%,55%,.05), hsla(${hue},84%,55%,.95))`;
  const visible = DATA.examples.filter(example => trueClass.value === 'all' || String(example.true_label) === trueClass.value);
  if (!visible.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Для выбранного класса нет примеров.';
    cards.append(empty);
    return;
  }
  for (const example of visible) {
    const card = document.createElement('article');
    const head = document.createElement('div');
    head.className = 'card-head';
    const title = document.createElement('h2');
    title.textContent = `${example.true_topic} · ${example.id}`;
    const subtitle = document.createElement('div');
    subtitle.className = 'card-subtitle';
    subtitle.textContent = `${example.tokens.length} токенов${example.truncated ? ' · обрезано' : ''}`;
    head.append(title, subtitle);

    const tokenBox = document.createElement('div');
    tokenBox.className = 'tokens';
    for (const token of example.tokens) {
      const span = document.createElement('span');
      span.className = 'token';
      span.textContent = token.text || '\u200b';
      const probability = token.probabilities[selectedIndex];
      span.style.backgroundColor = `hsla(${hue}, 84%, 55%, ${0.04 + probability * 0.88})`;
      span.title = probabilityTitle(token, token.probabilities);
      tokenBox.append(span);
    }

    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Метаданные генерации и исходный текст';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(example.metadata, null, 2);
    details.append(summary, pre);
    card.append(head, tokenBox, details);
    cards.append(card);
  }
}

highlight.addEventListener('change', render);
trueClass.addEventListener('change', render);
render();
</script>
</body>
</html>
"""
