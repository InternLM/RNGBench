"""
HTML template — per-turn replay viewer.

One frame per turn:
- left: messages_sent (with images/ASCII), the model reply, parse/env_result summary
- right: ground truth
- top: slider + metadata

Images reference the images/ directory next to game.json via relative paths. The
HTML file sits under <game_dir>/replay/, so image references use "../<path>".
"""

import html
import json
from typing import Any, Dict, List


def _prefix_image_paths(states: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    """Prefix each relative image_url.url so the HTML finds images relative to the replay/ dir."""
    import copy
    out = copy.deepcopy(states)
    for s in out:
        for m in s.get("messages_sent", []):
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = p.get("image_url", {}).get("url", "")
                    if url and not url.startswith(("data:", "http://", "https://", "/")):
                        p["image_url"]["url"] = prefix + url
    return out


def _prefix_gt_path(gt: Any, prefix: str) -> Any:
    if isinstance(gt, dict) and gt.get("type") == "image_url":
        import copy
        gt = copy.deepcopy(gt)
        url = gt.get("image_url", {}).get("url", "")
        if url and not url.startswith(("data:", "http://", "https://", "/")):
            gt["image_url"]["url"] = prefix + url
    return gt


def render_html(meta: Dict[str, Any], states: List[Dict[str, Any]], img_prefix: str = "../") -> str:
    states = _prefix_image_paths(states, img_prefix)
    gt = _prefix_gt_path(meta.get("ground_truth"), img_prefix)
    meta_for_js = {**meta, "ground_truth": gt}
    # Escape "</" so embedded JSON can't prematurely close the <script> tag.
    states_json = json.dumps(states, ensure_ascii=False).replace("</", "<\\/")
    meta_json = json.dumps(meta_for_js, ensure_ascii=False).replace("</", "<\\/")

    title = html.escape(meta.get("title", "replay"))
    mode = html.escape(meta.get("mode", ""))
    render_desc = html.escape(meta.get("render_desc", ""))
    grid = f"{meta.get('rows')}x{meta.get('cols')}"
    seed = meta.get("seed")
    rounds = meta.get("rounds_played")
    resp_count = meta.get("response_count")
    done = meta.get("done")
    subtitle = html.escape(meta.get("subtitle_scores", ""))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Replay · {title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: #f6f8fa; color: #1f2328; padding: 12px;
  }}
  .header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
  .header h1 {{ font-size: 1.1rem; color: #0969da; }}
  .header .meta {{ font-size: 0.78rem; color: #656d76; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; background: #0969da; color: #fff; margin-right: 4px; }}
  .badge.dual {{ background: #8250df; }}
  .badge.noaction {{ background: #cf222e; }}
  .controls {{ display: flex; gap: 8px; align-items: center; padding: 10px 0; flex-wrap: wrap; }}
  .controls input[type=range] {{ flex: 1; min-width: 300px; accent-color: #0969da; }}
  .controls .step-info {{ font-size: 0.8rem; color: #656d76; }}
  .controls button {{ padding: 4px 10px; border: 1px solid #d0d7de; border-radius: 4px; background: #fff; cursor: pointer; font-size: 0.8rem; }}
  .controls button:hover {{ background: #f3f4f6; }}
  .main {{ display: grid; grid-template-columns: minmax(0, 2.3fr) minmax(240px, 1fr); gap: 12px; margin-top: 8px; }}
  .panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; }}
  .panel-title {{ padding: 6px 12px; border-bottom: 1px solid #d0d7de; font-size: 0.78rem; font-weight: bold; color: #656d76; background: #f6f8fa; }}
  .panel-body {{ padding: 10px 12px; max-height: 80vh; overflow-y: auto; }}
  .turn-hdr {{ font-size: 0.85rem; padding: 6px 10px; background: #ddf4ff; color: #0550ae; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #0969da; }}
  .turn-hdr.A {{ background: #ffebe9; color: #86181d; border-left-color: #cf222e; }}
  .turn-hdr.B {{ background: #ddf4ff; color: #0550ae; border-left-color: #0969da; }}
  .turn-hdr.retry {{ background: #fff8c5; color: #7d4e00; border-left-color: #bf8700; }}
  .kv {{ font-size: 0.75rem; color: #656d76; margin-top: 4px; }}
  .kv b {{ color: #1f2328; }}
  .verdict.ok {{ color: #1a7f37; }}
  .verdict.bad {{ color: #cf222e; }}
  .msg {{ margin-bottom: 8px; padding: 8px 10px; border-radius: 6px; font-size: 0.78rem; line-height: 1.5; border-left: 3px solid transparent; word-break: break-word; white-space: pre-wrap; font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; }}
  .msg.system {{ background: #f5f0ff; border-left-color: #8250df; color: #5d3e9a; max-height: 100px; overflow: hidden; cursor: pointer; }}
  .msg.system.expanded {{ max-height: none; }}
  .msg.user {{ background: #ddf4ff; border-left-color: #0969da; color: #0550ae; }}
  .msg.assistant {{ background: #dafbe1; border-left-color: #1a7f37; color: #116329; }}
  .msg .role {{ display: block; font-size: 0.65rem; font-weight: bold; text-transform: uppercase; margin-bottom: 4px; opacity: 0.7; }}
  .msg pre.ascii {{ background: #fff; border: 1px solid #d0d7de; padding: 6px 8px; border-radius: 4px; margin: 4px 0; font-size: 0.7rem; color: #1f2328; overflow-x: auto; white-space: pre; }}
  .msg img {{ max-width: 260px; display: block; margin: 4px 0; border-radius: 4px; border: 1px solid #d0d7de; }}
  .resp-box {{ margin-top: 12px; padding: 8px 10px; border-radius: 6px; background: #fffbe8; border-left: 3px solid #bf8700; font-size: 0.78rem; white-space: pre-wrap; font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; color: #7d4e00; }}
  .reasoning {{ margin-top: 6px; padding: 6px 8px; border-radius: 4px; background: #f6f8fa; font-size: 0.72rem; color: #656d76; max-height: 120px; overflow-y: auto; cursor: pointer; }}
  .reasoning.expanded {{ max-height: none; }}
  .gt-img {{ max-width: 100%; border-radius: 6px; border: 1px solid #d0d7de; }}
  .gt-text {{ font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; font-size: 0.72rem; background: #f6f8fa; padding: 8px; border-radius: 4px; border: 1px solid #d0d7de; white-space: pre; overflow-x: auto; }}
  .keys {{ font-size: 0.7rem; color: #656d76; text-align: center; margin-top: 4px; }}
  kbd {{ background: #f6f8fa; padding: 1px 5px; border-radius: 3px; border: 1px solid #d0d7de; font-family: monospace; }}
  .empty {{ color: #9d9d9d; font-style: italic; }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="meta">
    <span class="badge {'dual' if meta.get('is_dual') else ''}">{mode}</span>
    <span>{render_desc} · grid {grid} · seed {seed} · rounds {rounds} · calls {resp_count} · done={done}</span>
  </div>
</div>
<div style="font-size:0.85rem;color:#1a7f37;font-weight:bold;">{subtitle}</div>

<div class="controls">
  <button id="btnFirst">|&lt;</button>
  <button id="btnPrev">&lt;</button>
  <input type="range" id="slider" min="0" max="{len(states)-1}" value="0">
  <button id="btnNext">&gt;</button>
  <button id="btnLast">&gt;|</button>
  <span class="step-info" id="stepInfo">0 / {max(len(states)-1,0)}</span>
</div>
<div class="keys"><kbd>&larr;</kbd>/<kbd>&rarr;</kbd> step &nbsp; <kbd>Home</kbd>/<kbd>End</kbd> jump</div>

<div class="main">
  <div class="panel">
    <div class="panel-title">Turn view (messages_sent + response)</div>
    <div class="panel-body" id="turnBody"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Ground Truth</div>
    <div class="panel-body" id="gtBody"></div>
  </div>
</div>

<script>
const STATES = {states_json};
const META = {meta_json};
const slider = document.getElementById('slider');
const stepInfo = document.getElementById('stepInfo');
const turnBody = document.getElementById('turnBody');
const gtBody = document.getElementById('gtBody');
let idx = 0;

function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function renderPart(p) {{
  if (!p || typeof p !== 'object') return '';
  if (p.type === 'text') {{
    const t = p.text || '';
    // detect a triple-backtick code block (ASCII board)
    const m = t.match(/^```\\n([\\s\\S]*?)\\n```$/);
    if (m) {{
      return '<pre class="ascii">' + esc(m[1]) + '</pre>';
    }}
    return '<span>' + esc(t) + '</span>';
  }}
  if (p.type === 'image_url') {{
    const url = p.image_url && p.image_url.url;
    if (!url) return '<span class="empty">[no image]</span>';
    return '<img src="' + esc(url) + '" loading="lazy">';
  }}
  return '<span class="empty">[' + esc(p.type || 'unknown') + ']</span>';
}}

function renderMsg(m) {{
  const role = m.role || 'user';
  const content = m.content;
  let body = '';
  if (typeof content === 'string') {{
    body = esc(content);
  }} else if (Array.isArray(content)) {{
    body = content.map(renderPart).join('');
  }}
  return '<div class="msg ' + role + '"><span class="role">' + role + '</span>' + body + '</div>';
}}

function renderTurn(s) {{
  const player = s.player;
  const hdrCls = player === 'A' ? 'A' : (player === 'B' ? 'B' : '');
  const retryBadge = s.retry_of ? ' <span style="color:#bf8700;">(retry of #' + s.retry_of + ')</span>' : '';
  let html = '';
  html += '<div class="turn-hdr ' + (s.call_type === 'retry' ? 'retry' : hdrCls) + '">';
  html += '#' + s.turn_id + ' · ' + esc(s.call_type);
  html += (player ? ' · Player ' + esc(player) : '');
  html += ' · round ' + s.round + retryBadge;
  html += '<div class="kv"><b>verdict:</b> <span class="verdict ' + (s.env_result && s.env_result.applied ? 'ok' : 'bad') + '">' + esc(s.verdict) + '</span>';
  if (s.parse && s.parse.coord) html += ' &nbsp; <b>coord:</b> ' + esc(s.parse.coord);
  if (s.env_result && s.env_result.phase_after) html += ' &nbsp; <b>phase_after:</b> ' + esc(s.env_result.phase_after);
  html += '</div></div>';

  // Messages sent
  html += '<div style="font-size:0.72rem;color:#656d76;margin:6px 0 4px;font-weight:bold;">Messages sent to model (' + s.messages_sent.length + '):</div>';
  html += s.messages_sent.map(renderMsg).join('');

  // Response
  const resp = s.response || {{}};
  html += '<div class="resp-box"><b>Model response:</b><br>' + esc(resp.content || '') + '</div>';
  if (resp.reasoning) {{
    html += '<div class="reasoning" onclick="this.classList.toggle(\\'expanded\\')"><b>Reasoning (click to expand):</b><br>' + esc(resp.reasoning) + '</div>';
  }}
  return html;
}}

function renderGT() {{
  const gt = META.ground_truth;
  if (!gt) {{ gtBody.innerHTML = '<span class="empty">(no ground truth)</span>'; return; }}
  if (gt.type === 'image_url') {{
    const url = gt.image_url && gt.image_url.url;
    gtBody.innerHTML = url ? '<img class="gt-img" src="' + esc(url) + '">' : '<span class="empty">[no image]</span>';
  }} else if (gt.type === 'text') {{
    const t = gt.text || '';
    const m = t.match(/^```\\n([\\s\\S]*?)\\n```$/);
    gtBody.innerHTML = '<pre class="gt-text">' + esc(m ? m[1] : t) + '</pre>';
  }} else {{
    gtBody.innerHTML = '<span class="empty">[' + esc(gt.type || 'unknown') + ']</span>';
  }}
}}

function goTo(i) {{
  idx = Math.max(0, Math.min(STATES.length - 1, i));
  slider.value = idx;
  stepInfo.textContent = idx + ' / ' + (STATES.length - 1);
  turnBody.innerHTML = renderTurn(STATES[idx]);
  // auto-expand system message toggles
  turnBody.querySelectorAll('.msg.system').forEach(el => {{
    el.onclick = () => el.classList.toggle('expanded');
  }});
}}

document.getElementById('btnFirst').onclick = () => goTo(0);
document.getElementById('btnLast').onclick = () => goTo(STATES.length - 1);
document.getElementById('btnPrev').onclick = () => goTo(idx - 1);
document.getElementById('btnNext').onclick = () => goTo(idx + 1);
slider.oninput = e => goTo(parseInt(e.target.value));
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') goTo(idx + 1);
  else if (e.key === 'ArrowLeft') goTo(idx - 1);
  else if (e.key === 'Home') goTo(0);
  else if (e.key === 'End') goTo(STATES.length - 1);
}});

renderGT();
goTo(0);
</script>
</body>
</html>
"""


def render_index_html(entries: List[Dict[str, Any]], title: str = "Replays Index") -> str:
    """entries = [{title, subtitle_scores, mode, render_desc, grid, seed, done, href}]"""
    rows = []
    for e in entries:
        status = "✓" if e.get("done") else "✗"
        href = html.escape(e["href"], quote=True)
        rows.append(f"""<tr data-href="{href}">
          <td>{status}</td>
          <td>{html.escape(e.get('title', ''))}</td>
          <td>{html.escape(e.get('mode', ''))}</td>
          <td>{html.escape(e.get('render_desc', ''))}</td>
          <td>{html.escape(e.get('grid', ''))}</td>
          <td>{e.get('seed', '')}</td>
          <td>{html.escape(e.get('subtitle_scores', ''))}</td>
          <td><a href="{href}" target="_blank" rel="noopener">open ↗</a></td>
        </tr>""")
    rows_html = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: #f6f8fa; padding: 16px; color: #1f2328; }}
h1 {{ font-size: 1.2rem; color: #0969da; margin-bottom: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; font-size: 0.85rem; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eaeef2; }}
th {{ background: #f6f8fa; color: #656d76; font-weight: bold; }}
tbody tr {{ cursor: pointer; }}
tbody tr:hover td {{ background: #ddf4ff; }}
tr:last-child td {{ border-bottom: none; }}
a {{ color: #0969da; text-decoration: none; font-weight: bold; }}
a:hover {{ text-decoration: underline; }}
</style>
</head><body>
<h1>{html.escape(title)} ({len(entries)} runs)</h1>
<table>
<thead><tr><th>done</th><th>label</th><th>mode</th><th>render</th><th>grid</th><th>seed</th><th>score</th><th></th></tr></thead>
<tbody id="tbody">
{rows_html}
</tbody></table>
<script>
// Row click navigates; clicks on anchors use their own href (and bubble-stop here so we don't double-navigate).
document.querySelectorAll('tbody tr[data-href]').forEach(tr => {{
  tr.addEventListener('click', e => {{
    if (e.target.closest('a')) return;  // let the anchor handle its own click
    window.location.href = tr.dataset.href;
  }});
}});
</script>
</body></html>
"""
