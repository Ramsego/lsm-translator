#!/usr/bin/env python3
"""Serve a sign-review folder with resume-from-disk and autosave.

The stock review.html keeps all progress in browser localStorage and only writes
review.csv when you click Download. Closing the tab (or opening it in a different
browser) loses everything since the last download -- which is how the first pass
on 57TvyH9902U was nearly lost.

This wrapper makes review.csv on disk the source of truth:
  * on GET /review.html it injects the saved verdicts into localStorage before
    the page's own script runs, so `advance()` lands on the first unrated clip
  * it monkey-patches localStorage.setItem to POST the state after every vote,
    signer change, or start/end mark, and writes review.csv immediately
  * it supports HTTP Range so scrubbing works (needed for the s/e marks)

Usage:  python3 review_server.py [clip_dir] [port]
"""

import csv, io, json, os, re, sys, threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

FIELDS = ["file", "word", "video_start", "verdict", "signer", "sign_start", "sign_end"]
STATE_FIELDS = ["verdict", "signer", "sign_start", "sign_end"]
_write_lock = threading.Lock()


def storage_key(clip_dir):
    return "signreview_" + os.path.basename(os.path.abspath(clip_dir))


def rows_from_html(clip_dir):
    """The canonical clip list is the ROWS literal baked into review.html."""
    html = open(os.path.join(clip_dir, "review.html"), encoding="utf-8").read()
    return json.loads(re.search(r"const ROWS = (\[.*?\]);", html, re.S).group(1))


def load_state(clip_dir):
    """Merge every review CSV present; later files win, non-empty values win."""
    state = {}
    for name in ("review.recovered.csv", "review.csv"):
        path = os.path.join(clip_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                entry = state.setdefault(row["file"], {})
                for f in STATE_FIELDS:
                    v = (row.get(f) or "").strip()
                    if v:
                        entry[f] = v
    return {k: v for k, v in state.items() if v}


def load_signers(clip_dir):
    """The interpreter quick-pick list. Registry file wins; if it doesn't exist
    yet, bootstrap from every signer string already used in review.csv, in
    first-appearance order, so today's session recovers immediately."""
    reg_path = os.path.join(clip_dir, "interpreters.json")
    if os.path.exists(reg_path):
        try:
            return json.load(open(reg_path, encoding="utf-8"))
        except Exception:
            pass
    seen = []
    for s in load_state(clip_dir).values():
        name = (s.get("signer") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def save_signers(clip_dir, names):
    names = [str(n).strip() for n in names if str(n).strip()]
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    target = os.path.join(clip_dir, "interpreters.json")
    tmp = target + ".tmp"
    with _write_lock:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(ordered, fh, ensure_ascii=False, indent=0)
        os.replace(tmp, target)
    return ordered


def save_state(clip_dir, state):
    rows = rows_from_html(clip_dir)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        s = state.get(r["file"], {})
        w.writerow({
            "file": r["file"], "word": r["word"], "video_start": r["video_start"],
            **{f: s.get(f, "") for f in STATE_FIELDS},
        })
    target = os.path.join(clip_dir, "review.csv")
    tmp = target + ".tmp"
    with _write_lock:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(buf.getvalue())
        os.replace(tmp, target)
    return sum(1 for s in state.values() if s.get("verdict"))


BANNER = """<div id="resumeBanner" style="position:sticky;top:0;z-index:10000;
  background:#0a3d0a;color:#baffba;font:bold 16px system-ui;padding:10px;
  text-align:center;border-bottom:2px solid #2fbf2f">%(banner)s</div>
"""

INJECT = """<script>
(function(){
  var KEY = %(key)s, SEED = %(seed)s;
  var SKEY = KEY + "_signers", SIGNERS = %(signers)s;

  var cur = {};
  try { cur = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
  // Merge PER FIELD, not per clip: a browser entry that carries only a verdict
  // must not blank out the signer / sign_start / sign_end held on disk.
  for (var f in SEED) {
    var d = SEED[f], c = cur[f] || (cur[f] = {});
    for (var k in d) if (!c[k]) c[k] = d[k];
  }
  localStorage.setItem(KEY, JSON.stringify(cur));

  // Interpreter quick-pick list: union of disk registry + whatever's already
  // in this browser (in case something was typed but not yet applied to a clip).
  var curSigners = [];
  try { curSigners = JSON.parse(localStorage.getItem(SKEY) || "[]"); } catch(e) {}
  var mergedSigners = SIGNERS.slice();
  for (var i2 = 0; i2 < curSigners.length; i2++) {
    if (mergedSigners.indexOf(curSigners[i2]) === -1) mergedSigners.push(curSigners[i2]);
  }
  localStorage.setItem(SKEY, JSON.stringify(mergedSigners));

  window.__lastSigner = mergedSigners.length ? mergedSigners[mergedSigners.length - 1] : "";

  var raw = localStorage.setItem.bind(localStorage);
  localStorage.setItem = function(k, v) {
    raw(k, v);
    if (k === KEY) {
      fetch("/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: v})
        .then(function(r){ return r.json(); })
        .then(function(d){
          var el = document.getElementById("autosave");
          if (el) el.textContent = "\\u25cf saved to review.csv \\u2014 " + d.reviewed + "/" + d.total;
        })
        .catch(function(){
          var el = document.getElementById("autosave");
          if (el) el.textContent = "\\u26a0 AUTOSAVE FAILED \\u2014 use the Download button";
        });
    } else if (k === SKEY) {
      fetch("/save_signers", {method: "POST", headers: {"Content-Type": "application/json"}, body: v})
        .then(function(r){ return r.json(); })
        .then(function(d){
          var el = document.getElementById("signersaved");
          if (el) el.textContent = "\\u25cf interpreters saved \\u2014 " + d.count;
        })
        .catch(function(){
          var el = document.getElementById("signersaved");
          if (el) el.textContent = "\\u26a0 interpreter list NOT saved to disk";
        });
    }
  };
  document.addEventListener("DOMContentLoaded", function(){
    var b = document.createElement("div");
    b.id = "autosave";
    b.style.cssText = "position:fixed;bottom:6px;right:10px;font:12px system-ui;color:#7ddf7d;opacity:.85;z-index:9999";
    b.textContent = "\\u25cf autosave on \\u2014 review.csv";
    document.body.appendChild(b);
    var s = document.createElement("div");
    s.id = "signersaved";
    s.style.cssText = "position:fixed;bottom:24px;right:10px;font:12px system-ui;color:#7ddf7d;opacity:.85;z-index:9999";
    s.textContent = "\\u25cf interpreters saved \\u2014 " + mergedSigners.length;
    document.body.appendChild(s);
    if (window.__lastSigner) {
      var h = document.createElement("div");
      h.style.cssText = "position:fixed;bottom:6px;left:10px;font:12px system-ui;color:#ffd54f;opacity:.85;z-index:9999";
      h.textContent = "known interpreters (" + mergedSigners.length + "): " + mergedSigners.join(" | ") + "  (press 1-9 or type to set)";
      document.body.appendChild(h);
    }
  });
})();
</script>
"""


class Handler(SimpleHTTPRequestHandler):
    clip_dir = "."

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/save":
            return self._handle_save()
        if path == "/save_signers":
            return self._handle_save_signers()
        return self.send_error(404)

    def _handle_save(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            state = json.loads(self.rfile.read(n) or "{}")
            reviewed = save_state(self.clip_dir, state)
            body = json.dumps({"ok": True, "reviewed": reviewed,
                               "total": len(rows_from_html(self.clip_dir))}).encode()
        except Exception as exc:                                  # surfaced in the badge
            body = json.dumps({"ok": False, "error": str(exc)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_save_signers(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            names = json.loads(self.rfile.read(n) or "[]")
            ordered = save_signers(self.clip_dir, names)
            body = json.dumps({"ok": True, "count": len(ordered)}).encode()
        except Exception as exc:
            body = json.dumps({"ok": False, "error": str(exc)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/review.html"):
            return self._serve_review(with_body=True)
        return self._serve_with_range()

    def do_HEAD(self):
        if self.path.split("?")[0] in ("/", "/review.html"):
            return self._serve_review(with_body=False)
        return super().do_HEAD()

    def _serve_review(self, with_body):
        html = open(os.path.join(self.clip_dir, "review.html"), encoding="utf-8").read()
        state = load_state(self.clip_dir)
        rows = rows_from_html(self.clip_dir)
        reviewed = sum(1 for s in state.values() if s.get("verdict"))
        next_idx = next(
            (i for i, r in enumerate(rows, 1)
             if not state.get(r["file"], {}).get("verdict")),
            len(rows) + 1,
        )
        banner = (
            "✓ RESUMING AT CLIP %d / %d — %d reviewed, confirmed by "
            "review_server.py from review.csv on disk" % (next_idx, len(rows), reviewed)
            if next_idx <= len(rows) else
            "✓ ALL %d CLIPS REVIEWED — confirmed by review_server.py from review.csv" % len(rows)
        )
        inject = INJECT % {
            "key": json.dumps(storage_key(self.clip_dir)),
            "seed": json.dumps(state),
            "signers": json.dumps(load_signers(self.clip_dir)),
        }
        html = html.replace("<script>", inject + "<script>", 1)
        # banner goes right after <body>, not wherever <script> happens to sit in the DOM
        html = re.sub(r"(<body[^>]*>)", r"\1" + (BANNER % {"banner": banner}), html, count=1)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def _serve_with_range(self):
        """Range support -- without it the video can't seek, so s/e marks break."""
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            self.wfile.write(fh.read(end - start + 1))

    def log_message(self, *a):
        pass


def already_serving(port):
    """True if something on this port is already answering as a review server."""
    import urllib.error, urllib.request
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/review.html" % port, timeout=2
        ) as r:
            return r.status == 200 and b"autosave" in r.read(200000)
    except Exception:
        return False


def main():
    clip_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8777
    url = "http://127.0.0.1:%d/review.html" % port

    if already_serving(port):
        print("A review server is ALREADY running on port %d." % port)
        print("Nothing to do -- just open:  %s" % url)
        print("To restart it:  pkill -f review_server.py && python3 %s %s %d"
              % (os.path.relpath(__file__), os.path.relpath(clip_dir), port))
        return

    state = load_state(clip_dir)
    total = len(rows_from_html(clip_dir))
    done = sum(1 for s in state.values() if s.get("verdict"))
    save_state(clip_dir, state)          # normalise review.csv on disk up front
    signers = save_signers(clip_dir, load_signers(clip_dir))  # bootstrap interpreters.json
    print("clip dir : %s" % clip_dir)
    print("resuming : %d/%d reviewed -> next is #%d" % (done, total, done + 1))
    print("interps  : %s" % (", ".join(signers) if signers else "(none yet)"))
    print("serving  : %s" % url)

    handler = partial(Handler, directory=clip_dir)
    Handler.clip_dir = clip_dir
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
    except OSError as exc:
        if exc.errno != 48:
            raise
        print("\nPort %d is taken by something that is not a review server." % port)
        print("Either free it:   lsof -nP -iTCP:%d -sTCP:LISTEN" % port)
        print("or pick another:  python3 %s %s %d"
              % (os.path.relpath(__file__), os.path.relpath(clip_dir), port + 1))
    except KeyboardInterrupt:
        print("\nstopped -- review.csv is up to date")


if __name__ == "__main__":
    main()
