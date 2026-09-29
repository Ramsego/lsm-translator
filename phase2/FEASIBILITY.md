# Phase 2 feasibility spike — findings

Tested end-to-end on one conference: **Z-pG_c9HoT8**, *Conferencia de prensa matutina, jueves 18 de
junio 2026* (a 10-minute section, 00:15:00–00:25:00, 1280×720 h264).

## Verdict: **GO** — for a hands + pose continuous-signing dataset. Face/non-manual capture is **not** viable from this source (see #2).

## The 4 unknowns

### 1. Interpreter video — ✓ (with a caveat)
Full conferences are on the official channel's **/streams** tab (~2 h each), with the LSM interpreter
in a **fixed bottom-right recuadro**, burned into the broadcast. Downloadable in sections with yt-dlp
(`--download-sections`, prefer h264 so OpenCV decodes it; the bundled ffmpeg 4.3 fails on
`--force-keyframes-at-cuts`, so omit it). Crop box for this video: **`X=1010 Y=460 W=270 H=260`**.

**Caveat — layout changes:** the recuadro is not static for the whole broadcast. In talking-head
segments the interpreter is reliably bottom-right (hands detected 98.7% over the first 120 s), but
sampled frames later in the section showed full-screen **slides** occupying that region. A
full-length pipeline must detect layout changes / track the interpreter region, not assume one fixed crop.

### 2. MediaPipe on the cropped interpreter — hands/pose ✓, **face ✗**
On the upscaled crop (120 s, scale 2):
- **hands: 98.7%**, **pose: 100%** — excellent.
- **face: 0.9%** — near-total failure. Upscaling does **not** help (scale 3 → 2.2%, scale 4 → 2.2%).

Probable causes: (a) in signing the **hands are constantly in front of the face**, occluding it; (b) the
PiP face is small/low-res and upscaling a blurry source adds no detail for the face-mesh detector.
**Implication:** Phase-2 mañanera data is effectively **hands + pose** (face rows present but mostly
NaN — dimensionally still fits the Phase-1 116-landmark schema). Non-manual face markers, a known
Phase-1 feature, cannot be recovered from this broadcast; declare as a limitation. (Worth one more
probe: run FaceLandmarker only on frames where the hands are lowered, to separate occlusion from
resolution.)

### 3. Estenográfica transcript — ✓ available, needs a real browser to fetch
Published per conference at a predictable gob.mx URL
(`.../version-estenografica-...-del-<d>-de-<mes>-de-<yyyy>`), ~27k words, speaker-tagged. **But**
gob.mx serves a JS bot-challenge — `urllib` and `cloudscraper` both got the 1.9 KB challenge page;
only WebFetch (full browser fetch) retrieved the real text. Production ingestion needs a **headless
browser (Playwright)** or a mirror. Content/format is confirmed fine; this is an engineering detail.

### 4. Audio → word alignment — ✓
faster-whisper `small` (CPU int8) produced clean Mexican-Spanish **word-level timestamps** (203 words
in 90 s), content coherent and matching the conference topic (administrative simplification). This is
the bridge: ASR timestamps anchor transcript words in video time; the interpreter signs them with a
short lag → approximate sign windows.

## Recommended full-Phase-2 shape (next plan)
1. **Transcript fetch** via Playwright (or a mirror); cache per date.
2. **Robust interpreter-region tracking** across the full 2 h (handle layout changes), not a static crop.
3. **Extract hands + pose** continuous landmarks (face deliberately dropped / best-effort).
4. **Audio ASR** word timestamps → align to estenográfica tokens → lagged candidate windows.
5. **Estimate the sign lag** empirically using word-bank-present words (e.g. numbers, *agua*, *pueblo*).
6. **Weak labels**: spotter (hands+arm, transcript-constrained) confirms which sign sits in each window;
   human verifies. Switch to the interpreter's own confirmed signs as references (within-signer).

## Scripts (this spike)
`download_mananera.py`, `locate_recuadro.py`, `extract_interpreter.py`, `fetch_estenografica.py`
(urllib version — blocked by the challenge; keep for the mirror path), `align_audio.py`.
Run from repo root. Video + transcripts live on the external drive / are gitignored.
