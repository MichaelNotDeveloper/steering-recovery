from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def write_examples_html(
    rows: Sequence[dict[str, Any]], path: str | Path
) -> Path:
    """Write a self-contained, filterable generation gallery."""

    if not rows:
        raise ValueError("cannot build an examples report without rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_rows = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Steering benchmark generations</title>
  <style>
    :root{color-scheme:light dark;--bg:#f3f5f8;--card:#fff;--text:#172033;--muted:#687386;--line:#d8dee8;--accent:#6d5ce7;--prompt:#fff1db;--generated:#e9f7ef}
    @media(prefers-color-scheme:dark){:root{--bg:#10141d;--card:#171d28;--text:#edf1f7;--muted:#aab3c2;--line:#303949;--accent:#a99cff;--prompt:#3a2d1e;--generated:#193829}}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}
    header{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:18px 24px}
    h1{font-size:22px;margin:0 0 12px}.controls{display:flex;flex-wrap:wrap;gap:10px}.control{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:12px}
    select{min-width:145px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text)}
    #count{align-self:flex-end;padding:8px 2px;color:var(--muted)}main{max-width:1200px;margin:0 auto;padding:22px;display:grid;gap:16px}
    article{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:17px;box-shadow:0 3px 14px #0000000a}
    .heading{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.heading h2{font-size:17px;margin:0}.badges{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
    .badge{padding:3px 8px;border-radius:999px;background:color-mix(in srgb,var(--accent) 14%,var(--card));color:var(--accent);font-size:12px;font-weight:600}
    pre.generation{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0;padding:14px;border:1px solid var(--line);border-radius:9px;font:13px/1.55 ui-monospace,monospace}
    .prompt{background:var(--prompt)}.generated{background:var(--generated)}.legend{color:var(--muted);font-size:12px}.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px}
    details{border-top:1px solid var(--line);padding-top:10px}summary{cursor:pointer;color:var(--accent);font-weight:600}.metadata{max-height:430px;overflow:auto;background:var(--bg);padding:12px;border-radius:8px;font-size:12px}
    .empty{text-align:center;color:var(--muted);padding:60px 20px}
  </style>
</head>
<body>
  <header>
    <h1>Steering benchmark generations</h1>
    <div class="controls">
      <label class="control">Method<select id="method"></select></label>
      <label class="control">Vector<select id="vector_slug"></select></label>
      <label class="control">Alpha<select id="alpha"></select></label>
      <label class="control">Source topic<select id="source_topic"></select></label>
      <div id="count"></div>
    </div>
  </header>
  <main id="gallery"></main>
  <script id="examples-data" type="application/json">__EXAMPLES_DATA__</script>
  <script>
    const rows=JSON.parse(document.getElementById('examples-data').textContent);
    const filters=[
      {id:'method',label:'All methods',sort:false},
      {id:'vector_slug',label:'All vectors',sort:false},
      {id:'alpha',label:'All alpha values',sort:true},
      {id:'source_topic',label:'All source topics',sort:false}
    ];
    const selections={};
    for(const filter of filters){
      const select=document.getElementById(filter.id);selections[filter.id]=select;
      select.append(new Option(filter.label,''));
      let values=[...new Set(rows.map(row=>String(row[filter.id])))];
      values.sort(filter.sort?(a,b)=>Number(a)-Number(b):(a,b)=>a.localeCompare(b));
      for(const value of values)select.append(new Option(value,value));
      select.addEventListener('change',render);
    }
    function badge(text){const item=document.createElement('span');item.className='badge';item.textContent=text;return item}
    function render(){
      const visible=rows.filter(row=>filters.every(filter=>!selections[filter.id].value||String(row[filter.id])===selections[filter.id].value));
      document.getElementById('count').textContent=`${visible.length} / ${rows.length} generations`;
      const gallery=document.getElementById('gallery');gallery.replaceChildren();
      if(!visible.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No generations match these filters.';gallery.append(empty);return}
      for(const row of visible){
        const article=document.createElement('article');
        const heading=document.createElement('div');heading.className='heading';
        const title=document.createElement('h2');title.textContent=`${row.vector_name} · ${row.method} · α=${row.alpha}`;heading.append(title);article.append(heading);
        const badges=document.createElement('div');badges.className='badges';
        badges.append(badge(`source: ${row.source_topic}`),badge(`target p: ${Number(row.target_probability).toFixed(4)}`));
        for(const order of [1,2,3])if(row[`distinct_${order}`]!==undefined)badges.append(badge(`Dist-${order}: ${Number(row[`distinct_${order}`]).toFixed(4)}`));
        if(row.slor!==undefined)badges.append(badge(`SLOR: ${Number(row.slor).toFixed(4)}`));
        badges.append(badge(`seed: ${row.seed}`));article.append(badges);
        const legend=document.createElement('div');legend.className='legend';legend.innerHTML='<span class="swatch prompt"></span>prompt <span class="swatch generated"></span>generated';article.append(legend);
        const text=document.createElement('pre');text.className='generation';
        const prompt=document.createElement('span');prompt.className='prompt';prompt.textContent=row.prompt_text;
        const generated=document.createElement('span');generated.className='generated';generated.textContent=row.generated_text;text.append(prompt,generated);article.append(text);
        const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='All generation metadata';
        const metadata=document.createElement('pre');metadata.className='metadata';metadata.textContent=JSON.stringify(row,null,2);details.append(summary,metadata);article.append(details);gallery.append(article);
      }
    }
    render();
  </script>
</body>
</html>"""
    path.write_text(template.replace("__EXAMPLES_DATA__", encoded_rows), encoding="utf-8")
    return path
