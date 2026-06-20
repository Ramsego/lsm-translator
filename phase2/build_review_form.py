"""
Generate a self-contained review.html for a clip folder (from make_review_clips.py).

Open the generated review.html in a browser: it plays each clip on loop, shows the
word, sentence context (target highlighted), and gives Yes / No / Negation / Skip / Back
buttons + "mark sign moment" (captures the clip's current time). Progress auto-saves to
localStorage (resume anytime). Backspace goes back to the previous clip to fix mistakes.
When done, "Download review.csv" exports verdicts + sign times.

Verdicts:
  y   — sign is here as expected → gold_labels
  n   — sign doesn't appear (signer conveyed implicitly) → implicit_signs
  neg — negated form is here (different sign, e.g. no-tener) → gold_labels under word:neg
  ''  — skip (come back later)

Usage:
    python phase2/build_review_form.py --dir <review/VIDEOID>
Then: open <review/VIDEOID>/review.html in Chrome/Safari, review, Download review.csv
(overwrite the one in the folder), then run collect_gold.py.
"""

import argparse
import csv
import json
from pathlib import Path

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Sign review</title>
<style>
 body{{font-family:system-ui;margin:0;background:#111;color:#eee;text-align:center}}
 #word{{font-size:2.2em;margin:.3em;font-weight:700}}
 #ctx{{color:#bbb;font-size:.95em;max-width:750px;margin:.2em auto .5em;line-height:1.6;
       min-height:1.4em;word-break:break-word}}
 video{{max-height:55vh;background:#000;border-radius:8px}}
 #bar{{margin:.5em;font-size:1.1em;color:#aaa}}
 button{{font-size:1.1em;padding:.6em 1.4em;margin:.3em;border:0;border-radius:8px;cursor:pointer}}
 .yes{{background:#2e7d32;color:#fff}} .no{{background:#c62828;color:#fff}}
 .neg{{background:#6a1b9a;color:#fff}} .skip{{background:#555;color:#fff}}
 .mark{{background:#1565c0;color:#fff}} .back{{background:#37474f;color:#fff}}
 .dl{{background:#f9a825;color:#000}}
 #marked{{color:#64b5f6;min-height:1.2em}}
</style></head><body>
<div id="bar"></div>
<div id="signer" style="font-size:1.3em;color:#ffd54f">signer 1</div>
<div id="word"></div>
<div id="ctx"></div>
<video id="vid" autoplay loop controls></video>
<div id="marked"></div>
<div>
 <button class="mark" onclick="markStart()">Sign START (s)</button>
 <button class="mark" onclick="markEnd()">Sign END (e)</button>
</div>
<div>
 <button class="yes" onclick="vote('y')">Yes — sign here (→)</button>
 <button class="no" onclick="vote('n')">Doesn't appear (↓)</button>
 <button class="neg" onclick="vote('neg')">Negated form (g)</button>
 <button class="skip" onclick="vote('')">Skip</button>
 <button class="back" onclick="back()">← Back (Backspace)</button>
</div>
<div style="color:#aaa">interpreter (press a number 1–9 when the person changes):
 <button class="skip" onclick="setSigner('1')">1</button>
 <button class="skip" onclick="setSigner('2')">2</button>
 <button class="skip" onclick="setSigner('3')">3</button>
 <button class="skip" onclick="setSigner('4')">4</button>
 <button class="skip" onclick="setSigner('5')">5</button>
 <button class="skip" onclick="setSigner('6')">6</button>
</div>
<div><button class="dl" onclick="download()">⬇ Download review.csv</button></div>
<script>
const ROWS = {rows};
const KEY = "signreview_{vid}";
let state = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let i = 0;
let lastSigner = "1";
const vid = document.getElementById('vid');

function setSigner(n){{
  lastSigner = n;
  const r=ROWS[i]; if(r){{ state[r.file]=state[r.file]||{{}}; state[r.file].signer=n;
    localStorage.setItem(KEY,JSON.stringify(state)); }}
  document.getElementById('signer').textContent='signer '+n;
}}

function renderCtx(ctx){{
  if(!ctx){{ document.getElementById('ctx').innerHTML=''; return; }}
  document.getElementById('ctx').innerHTML = ctx.replace(
    /\*\*(.+?)\*\*/g,
    '<strong style="color:#ffd54f;background:rgba(255,213,79,.12);padding:0 3px;border-radius:3px">$1</strong>'
  );
}}

function showCurrent(){{
  if(i>=ROWS.length){{ document.getElementById('word').textContent='✅ Done — Download review.csv'; vid.src=''; renderCtx(''); return; }}
  if(i<0) i=0;
  const r=ROWS[i];
  const reviewed=Object.values(state).filter(s=>s.verdict!=='').length;
  document.getElementById('bar').textContent=`Clip ${{i+1}} / ${{ROWS.length}}  ·  reviewed ${{reviewed}}`;
  document.getElementById('word').textContent=r.word.toUpperCase();
  renderCtx(r.context||'');
  vid.src=encodeURI(r.file);
  const sg=(state[r.file]&&state[r.file].signer)||lastSigner;
  document.getElementById('signer').textContent='signer '+sg;
  showWin();
}}

function advance(){{
  while(i<ROWS.length && state[ROWS[i].file] && state[ROWS[i].file].verdict!=='') i++;
  showCurrent();
}}

function back(){{
  if(i>0) i--;
  showCurrent();
}}

function vote(v){{
  const r=ROWS[i];
  state[r.file]=state[r.file]||{{}};
  state[r.file].verdict=v;
  if(!state[r.file].signer) state[r.file].signer=lastSigner;
  localStorage.setItem(KEY,JSON.stringify(state));
  i++; advance();
}}

function showWin(){{
  if(i>=ROWS.length) return;
  const s=state[ROWS[i].file]||{{}};
  let t='';
  if(s.verdict) t+='['+s.verdict+'] ';
  if(s.sign_start!=null) t+='start '+s.sign_start+'s ';
  if(s.sign_end!=null) t+='→ end '+s.sign_end+'s';
  document.getElementById('marked').textContent=t;
}}

function markStart(){{
  const r=ROWS[i]; state[r.file]=state[r.file]||{{}};
  state[r.file].sign_start=(parseFloat(r.video_start)+vid.currentTime).toFixed(1);
  localStorage.setItem(KEY,JSON.stringify(state)); showWin();
}}

function markEnd(){{
  const r=ROWS[i]; state[r.file]=state[r.file]||{{}};
  state[r.file].sign_end=(parseFloat(r.video_start)+vid.currentTime).toFixed(1);
  localStorage.setItem(KEY,JSON.stringify(state)); showWin();
}}

function download(){{
  let out="file,word,video_start,verdict,signer,sign_start,sign_end\\n";
  for(const r of ROWS){{
    const s=state[r.file]||{{}};
    out+=`${{r.file}},${{r.word}},${{r.video_start}},${{s.verdict||''}},${{s.signer||''}},${{s.sign_start||''}},${{s.sign_end||''}}\\n`;
  }}
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([out],{{type:'text/csv'}}));
  a.download='review.csv'; a.click();
}}

document.addEventListener('keydown',e=>{{
  if(e.key==='ArrowRight') vote('y');
  else if(e.key==='ArrowDown') vote('n');
  else if(e.key==='g'){{ e.preventDefault(); vote('neg'); }}
  else if(e.key==='s'){{ e.preventDefault(); markStart(); }}
  else if(e.key==='e'){{ e.preventDefault(); markEnd(); }}
  else if(e.key==='Backspace'){{ e.preventDefault(); back(); }}
  else if(e.key>='1'&&e.key<='9') setSigner(e.key);
}});

advance();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Folder containing review.csv + clips.")
    args = ap.parse_args()

    rows = []
    with open(args.dir / "review.csv") as f:
        for r in csv.DictReader(f):
            rows.append({"file": r["file"], "word": r["word"],
                         "video_start": r["video_start"],
                         "context": r.get("context", "")})

    out = args.dir / "review.html"
    out.write_text(HTML.format(rows=json.dumps(rows, ensure_ascii=False), vid=args.dir.name))
    print(f"Wrote {out}  ({len(rows)} clips)")
    print("Keys: → = Yes, ↓ = Doesn't appear, g = Negated form, Backspace = Back, s = sign start, e = sign end.")
    print("When done, click 'Download review.csv', overwrite the one in this folder, then run collect_gold.py.")


if __name__ == "__main__":
    main()
