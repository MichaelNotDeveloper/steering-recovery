from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from steering_recovery.steering.epistemic.statistics import METRIC_LABELS


def write_epistemic_examples_html(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Write a self-contained gallery with selectable token statistics."""

    if not rows:
        raise ValueError("at least one epistemic example is required")
    payload = {
        "rows": [dict(row) for row in rows],
        "metrics": dict(METRIC_LABELS),
        "metadata": dict(metadata),
    }
    serialized = (
        json.dumps(payload, ensure_ascii=False)
        .replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        _HTML_TEMPLATE.replace("__REPORT_PAYLOAD__", serialized), encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return output_path


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Epistemic steering · MC dropout</title>
  <style>
    :root{color-scheme:light;--ink:#172033;--muted:#647084;--line:#dbe2ec;--panel:#fff;--canvas:#f4f6fa;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
    *{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink)}main{width:min(1220px,calc(100% - 32px));margin:38px auto 70px}
    h1{margin:0 0 8px;font-size:clamp(28px,4vw,43px);letter-spacing:-.04em}.lead{max-width:850px;margin:0;color:var(--muted);line-height:1.55}
    .summary{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.pill{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:white;font-size:13px}
    .toolbar{position:sticky;top:10px;z-index:4;display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin:22px 0;padding:16px;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.95);box-shadow:0 10px 30px rgba(41,55,78,.08);backdrop-filter:blur(10px)}
    label{display:grid;gap:6px;color:var(--muted);font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.06em}select{width:100%;padding:10px 11px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;color:var(--ink);font:inherit}
    .legend{grid-column:1/-1;display:grid;grid-template-columns:auto 1fr auto;gap:9px;align-items:center;color:var(--muted);font-size:12px}.gradient{height:11px;border-radius:999px;background:linear-gradient(90deg,#eef2ff,#4f46e5)}
    #gallery{display:grid;gap:15px}.card{padding:20px;border:1px solid var(--line);border-radius:16px;background:var(--panel);box-shadow:0 7px 23px rgba(33,48,72,.045)}
    .card-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin-bottom:14px}.card h2{margin:0;font-size:18px}.sub{color:var(--muted);font-size:13px}.prompt{margin:0 0 12px;color:#4b5563;line-height:1.65}.tokens{font:14px/2.1 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.token{padding:2px 0;border-radius:4px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
    details{margin-top:14px;color:var(--muted);font-size:13px}summary{cursor:pointer;font-weight:700;color:#46536a}pre{max-height:350px;overflow:auto;padding:13px;border-radius:10px;background:#f7f9fc;color:#334155;font-size:12px;white-space:pre-wrap}.empty{padding:38px;text-align:center;color:var(--muted);background:#fff;border:1px dashed #cbd5e1;border-radius:15px}
    @media(max-width:820px){.toolbar{position:static;grid-template-columns:1fr 1fr}}@media(max-width:520px){.toolbar{grid-template-columns:1fr}}
  </style>
</head>
<body><main>
  <header><h1>MC-dropout epistemic steering</h1><p class="lead">Токены окрашены по статистике 20 dropout-предсказаний денойзера для соответствующего steered hidden. Цвет нормируется по устойчивому диапазону 5–95% выбранной статистики.</p><div class="summary" id="summary"></div><details><summary>Метаданные эксперимента</summary><pre id="globalMetadata"></pre></details></header>
  <section class="toolbar" aria-label="Epistemic report controls">
    <label>σ денойзера<select id="sigma"></select></label><label>Steering-вектор<select id="vector"></select></label><label>Сила α<select id="alpha"></select></label><label>Статистика<select id="metric"></select></label>
    <div class="legend"><span id="low">low</span><div class="gradient"></div><span id="high">high</span></div>
  </section>
  <section id="gallery"></section>
</main><script>
const DATA=__REPORT_PAYLOAD__;
const controls={sigma:document.querySelector('#sigma'),vector:document.querySelector('#vector'),alpha:document.querySelector('#alpha'),metric:document.querySelector('#metric')};
const gallery=document.querySelector('#gallery');
const unique=(values)=>[...new Set(values)];
const sigmas=unique(DATA.rows.map(row=>String(row.sigma))).sort((a,b)=>Number(a)-Number(b));
const vectors=unique(DATA.rows.map(row=>row.vector_slug));
const alphas=unique(DATA.rows.map(row=>String(row.alpha))).sort((a,b)=>Number(a)-Number(b));
const vectorNames=new Map(DATA.rows.map(row=>[row.vector_slug,row.vector_name]));
function addOptions(select,values,labeler=value=>value){for(const value of values){const option=document.createElement('option');option.value=value;option.textContent=labeler(value);select.append(option)}}
addOptions(controls.sigma,sigmas,value=>`σ = ${value}`);addOptions(controls.vector,vectors,value=>vectorNames.get(value));addOptions(controls.alpha,alphas,value=>`α = ${value}`);addOptions(controls.metric,Object.keys(DATA.metrics),value=>DATA.metrics[value]);
document.querySelector('#globalMetadata').textContent=JSON.stringify(DATA.metadata,null,2);
for(const value of [`${DATA.rows.length} примеров`,`${sigmas.length} denoisers`,`${vectors.length} steering-вектора`,`${DATA.metadata.mc_samples} MC-прогонов`]){const pill=document.createElement('span');pill.className='pill';pill.textContent=value;document.querySelector('#summary').append(pill)}
function quantile(values,q){const sorted=[...values].sort((a,b)=>a-b);const position=(sorted.length-1)*q;const base=Math.floor(position);const rest=position-base;return sorted[base+1]===undefined?sorted[base]:sorted[base]+rest*(sorted[base+1]-sorted[base])}
function tokenTitle(token){const lines=[`step ${token.step} · token id ${token.token_id}`,JSON.stringify(token.token_text)];for(const [key,label] of Object.entries(DATA.metrics)){lines.push(`${label}: ${Number(token[key]).toPrecision(6)}`)}return lines.join('\n')}
function render(){gallery.replaceChildren();const metric=controls.metric.value;const allValues=DATA.rows.flatMap(row=>row.token_statistics.map(token=>Number(token[metric])));let low=quantile(allValues,.05),high=quantile(allValues,.95);if(!(high>low)){low=Math.min(...allValues);high=Math.max(...allValues)+1e-12}document.querySelector('#low').textContent=low.toPrecision(4);document.querySelector('#high').textContent=high.toPrecision(4);
  const visible=DATA.rows.filter(row=>String(row.sigma)===controls.sigma.value&&row.vector_slug===controls.vector.value&&String(row.alpha)===controls.alpha.value);
  if(!visible.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='Для выбранного условия нет примеров.';gallery.append(empty);return}
  for(const row of visible){const card=document.createElement('article');card.className='card';const head=document.createElement('div');head.className='card-head';const title=document.createElement('h2');title.textContent=`${row.source_topic} · ${row.sample_id}`;const sub=document.createElement('div');sub.className='sub';sub.textContent=`${row.vector_name} · α=${row.alpha} · σ=${row.sigma}`;head.append(title,sub);
    const prompt=document.createElement('p');prompt.className='prompt';prompt.textContent=`Prompt: ${row.prompt_text}`;const tokens=document.createElement('div');tokens.className='tokens';for(const token of row.token_statistics){const span=document.createElement('span');span.className='token';span.textContent=token.token_text||'\u200b';const normalized=Math.max(0,Math.min(1,(Number(token[metric])-low)/(high-low)));span.style.backgroundColor=`hsla(245,78%,57%,${.04+.88*normalized})`;span.title=tokenTitle(token);tokens.append(span)}
    const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Все метаданные генерации';const pre=document.createElement('pre');pre.textContent=JSON.stringify(row.metadata,null,2);details.append(summary,pre);card.append(head,prompt,tokens,details);gallery.append(card)}}
for(const select of Object.values(controls)){select.addEventListener('change',render)}render();
</script></body></html>
"""
