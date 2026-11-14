# Dashboard redesign and test-board demo: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A clean, professional dashboard for the station, plus a demo mode so anyone can try it on held-out test boards or an uploaded photo. It runs locally with one Docker command and online as a private Hugging Face Space.

**Architecture:**
- **Server (FastAPI).**
  - New endpoints serve the test boards, uploads, plain board images and shift statistics.
  - Inspections keep running one at a time through the existing `Station.lock`.
  - Uploads are inspected with `log=False`. They are held in memory only and never written to the store.
- **Frontend.**
  - Vanilla JS, rewritten as `static/index.html` + `static/app.css` + `static/app.js` + `static/labels.js`, with bundled fonts and no CDN.
  - Balloons are drawn in the browser as SVG over the plain image.
- **Demo image.** One Dockerfile builds both the local demo and the Space. It pulls the models and demo boards from the private HF repo using a build secret.

**Tech Stack:** Python 3.12/3.13, FastAPI, SQLite, OpenCV, ExecuTorch 1.6.0, vanilla JS/CSS, Docker, Hugging Face Spaces (Docker SDK), pytest, Chrome DevTools MCP for visual checks.

**Spec:** `docs/superpowers/specs/2026-11-14-dashboard-demo-design.md`. Read it first; colours, components and copy come from there.

## Global Constraints

- **Offline:** no network fetch from the page (no CDN, no Google Fonts). Fonts are bundled in `pcbinspect/static/fonts/`: Archivo 400/600/800 and Archivo Narrow 500/700, as woff2, SIL OFL.
- **Colours, exactly:**
  - bench `#EEF1F0`; panel `#FFFFFF`; ink `#16201C`; secondary text `#5E6A66`; line `#DCE3EE`;
  - blue `#1D3F8F` / `#9DB0DA` / `#EEF2FB`; sidebar `#14213F`, selected `#25386A`;
  - PASS `#2E7D32`, CHECK `#C98A0B` (lamp `#F2B12B`), FAIL `#C62828`.
- **Verdict colours are only for verdicts and reject/OK pills.**
- **Demo mode:** only when `demo: true`. With it off, `/api/samples*` and `/api/inspect/upload` return 404.
- **Uploads:**
  - JPEG or PNG, checked by decoding; at most 10 MB and 4000 px on a side.
  - Never written to the store or the image store.
  - Verdict always REVIEW, with the reason "not calibrated for this board".
  - Sizes in pixels.
- **Known-answer match:** IoU ≥ 0.1, or the finding's centre inside the marked box. This is the same rule as `training/eval_golden.py:40`; reuse `training.eval_bare.iou` logic, but do not import `training` from `pcbinspect` (copy the 6-line IoU into `pcbinspect/demo.py`).
- **Online disclosure line** (exact): "Demo on test boards. Online copy runs on a shared cloud CPU, not the Raspberry Pi; Pi timings are in the technical report."
- **Secrets:** the HF token is never written into the image, a layer, a file in git or a log. Use the Docker build secret `HF_TOKEN` only.
- **Commits:** local only, authored by the user, with no co-author line.
- **Existing tests:** all 312 must keep passing. Run them with `C:/Users/sulai/pcb_work/venv/Scripts/python.exe -m pytest -q`.
- **Deviation from the spec, noted:** the frontend is four static files instead of one HTML file. It is still offline and framework-free; `/static` is already mounted (`pcbinspect/app.py:781`).

## Review Focus

1. **An odd but valid upload** (grayscale JPEG, RGBA PNG, 16-bit PNG) is inspected rather than crashing. A corrupt or non-image file gets a clear 400. These are pinned in Task 4.
2. **No demo boards present** (the `demo/` folder is missing or empty): `/api/samples` returns `[]` and the page shows "No test boards installed", not a crash. Pinned in Task 2 and Task 7.
3. **An unknown or expired upload token** returns 404, not another visitor's result. Pinned in Task 4.
4. **Stats on an empty log** return zero counts and `null` medians, with no NaN in the JSON. Pinned in Task 1.
5. **Pressing Inspect again while Llama is still writing advice** for the previous board returns the new verdict without waiting for the advice. Pinned in Task 3.

---

## File structure

| File | Responsibility |
|---|---|
| `pcbinspect/store.py` (modify) | `stats(since_ts)`, `reset()` |
| `pcbinspect/demo.py` (create) | Test-board catalogue (`Sample`, `load_samples`), known-answer matching (`match_known`), upload decoding (`decode_upload`), nightly reset timing (`seconds_until`) |
| `pcbinspect/app.py` (modify) | Station: `_run_inspection` (shared core), `inspect_sample`, `inspect_upload`, an upload result cache, a queue counter, the nightly reset thread; routes for samples, upload, images and stats |
| `pcbinspect/static/index.html` (rewrite) | Page skeleton: sidebar, header, one `<section>` per screen |
| `pcbinspect/static/app.css` (create) | Tokens and components from the spec |
| `pcbinspect/static/app.js` (create) | Screens, API calls, balloons, grouping of advice |
| `pcbinspect/static/labels.js` (create) | Plain-wording map and display names, kept in one place |
| `pcbinspect/static/fonts/Archivo*.woff2` (add) | Bundled fonts |
| `config/app.yaml` (modify) | `demo: false`, `upload_design: deeppcb`, `samples_per_design: 4`, `shift_hours: 8`, `demo_reset_hour: 3` |
| `deploy/Dockerfile.demo`, `deploy/DEMO.md` (create) | Demo image and how to run it locally and on the Space |
| `tests/test_store.py`, `tests/test_demo.py`, `tests/test_app_demo.py` | Tests |

---

### Task 1: Shift statistics and reset in the store

**Files:**
- Modify: `pcbinspect/store.py` (after `recent`, line ~222)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces:
  - `Store.stats(since_ts: float) -> dict`, with keys `boards, passed, review, failed` (int) and `verdict_ms_median, advice_ms_median` (float | None).
  - `Store.reset() -> None`: deletes every row from inspections, defects, advice, feedback_events and alerts, and every image in the image store.
  - designs, model_versions, eyetests and customer_rules are kept.

- [ ] **Step 1: Write the failing tests**

```python
def test_stats_counts_and_medians(tmp_path):
    s = Store(tmp_path)
    for v, ms in [("PASS", 100.0), ("REVIEW", 300.0), ("FAIL", 200.0), ("PASS", 400.0)]:
        insp = make_insp(v); insp.latency_ms = {"total_to_verdict": ms}
        iid = s.log_inspection(insp, None)
        s.log_advice(iid, "template", None, {"summary": "x", "steps": []}, True, None, 5000.0, None, None)
    st = s.stats(0.0)
    assert (st["boards"], st["passed"], st["review"], st["failed"]) == (4, 2, 1, 1)
    assert st["verdict_ms_median"] == 250.0 and st["advice_ms_median"] == 5000.0

def test_stats_on_an_empty_log_has_no_nan(tmp_path):
    st = Store(tmp_path).stats(0.0)
    assert st == {"boards": 0, "passed": 0, "review": 0, "failed": 0,
                  "verdict_ms_median": None, "advice_ms_median": None}

def test_reset_clears_inspections_but_keeps_designs(tmp_path):
    s = Store(tmp_path)
    s.upsert_design("d", "bare")
    sha = s.put_image(b"img")
    s.log_inspection(make_insp(), sha)
    s.reset()
    assert s.recent() == [] and s.design("d") is not None and not s.image_path(sha).exists()
```

- [ ] **Step 2: Run them; expect FAIL** (`AttributeError: 'Store' object has no attribute 'stats'`):
  `C:/Users/sulai/pcb_work/venv/Scripts/python.exe -m pytest tests/test_store.py -q`
- [ ] **Step 3: Implement `stats` and `reset`.**
  - The verdict time is `latency_json` → `total_to_verdict`.
  - The medians use `statistics.median` over non-null values; an empty list gives `None`.
  - Run `reset` in one transaction, then unlink the image files.
- [ ] **Step 4: Run `tests/test_store.py`; expect all PASS.**
- [ ] **Step 5: Commit:** `git add pcbinspect/store.py tests/test_store.py && git commit -m "Store: shift statistics and a demo reset"`

### Task 2: Test-board catalogue, known-answer matching, upload decoding

**Files:**
- Create: `pcbinspect/demo.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Produces:
  - `@dataclass Sample: file: str; design: str; kind: str ("bare"|"assembled"); good: bool; boxes: list[list[int]]; has_golden: bool`.
  - `load_samples(root: Path, per_design: int = 4) -> list[Sample]` reads `root/"demo"/"manifest.json"`.
    - It keeps only entries whose image exists.
    - Per design it takes up to `per_design` entries, the defective first and then a good one if any; the order is stable.
    - kind is `"assembled"` when the design starts with `visa_`.
    - A missing manifest gives `[]`.
  - `match_known(found: list[tuple[int,int,int,int]], known: list[list[int]]) -> dict` returns `{"marked": int, "found": int, "extra": int}`.
    - A marked box is found when any finding has IoU ≥ 0.1 with it, or the finding's centre lies inside it.
    - extra = findings that match no marked box.
  - `decode_upload(data: bytes, max_bytes: int = 10_000_000, max_side: int = 4000) -> np.ndarray` returns BGR uint8.
    - It raises `UploadError(status: int, message: str)`: 413 when too big; 400 when it doesn't decode or the format is not JPEG/PNG (check magic bytes `\xff\xd8\xff` / `\x89PNG`); 400 when a side exceeds `max_side`.
    - Grayscale is converted to 3 channels, RGBA drops alpha, and 16-bit is scaled to 8-bit.
  - `seconds_until(hour: int, now: datetime) -> float` returns the seconds until the next `hour:00` local time, always > 0.

- [ ] **Step 1: Write the failing tests** (`tests/test_demo.py`):
  - `test_load_samples_picks_defective_then_good_per_design`: a manifest with 3 deeppcb defective, 2 deeppcb good and 1 `visa_pcb1` good, files created as tiny JPEGs. With `per_design=2`, it returns deeppcb [defective, good] and visa [good] with `kind == "assembled"`.
  - `test_load_samples_skips_missing_files_and_missing_manifest`: returns `[]` with no manifest; an entry whose file is absent is skipped.
  - `test_match_known_counts_found_and_extra`: known `[[0,0,10,10],[50,50,60,60]]`, found `[(2,2,12,12),(100,100,110,110)]` gives `{"marked":2,"found":1,"extra":1}`. A finding whose centre is inside a box but has IoU < 0.1 (a `(4,4,6,6)` finding in `[0,0,40,40]`) counts as found.
  - `test_decode_upload_accepts_gray_rgba_and_16bit_png`: each comes back with shape `(h, w, 3)` and dtype `uint8`.
  - `test_decode_upload_rejects_bad_input`: `b"hello"` gives 400; 10 MB + 1 bytes of a valid header gives 413; a 4001×10 PNG gives 400; a valid GIF gives 400.
  - `test_seconds_until_next_reset`: `now = 2026-11-14 02:30` gives 1800; `now = 03:00:00` gives 86400.
- [ ] **Step 2: Run them; expect FAIL** (`ModuleNotFoundError: pcbinspect.demo`).
- [ ] **Step 3: Implement `pcbinspect/demo.py`.** Decode with `cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)`.
- [ ] **Step 4: Run `tests/test_demo.py`; expect all PASS.**
- [ ] **Step 5: Commit:** `git commit -m "Demo: test-board catalogue, known-answer matching, safe upload decoding"`

### Task 3: Server: test boards, plain images, stats, the queue counter, demo flag

**Files:**
- Modify: `pcbinspect/app.py`. `Station.inspect` (lines 208–243) becomes a thin wrapper; `InspectReq` (line 716); `create_app` routes; `/api/state` (line 787).
- Modify: `config/app.yaml`. Add `demo: false`, `upload_design: deeppcb`, `samples_per_design: 4`, `shift_hours: 8`, `demo_reset_hour: 3`.
- Test: `tests/test_app_demo.py`

**Interfaces:**
- Consumes: `load_samples`, `match_known`, `Sample` (Task 2); `Store.stats` (Task 1).
- Produces:
  - **`Station._run_inspection`.** Signature: `Station._run_inspection(bgr, design, alpha, board_id, golden, force_review=None, capture_info=None) -> int` (returns iid). It is the existing body of `inspect` from `with self.lock:` to queueing advice. `inspect()` keeps its signature and behaviour and calls it.
  - **`Station.waiting: int`.** Incremented before taking `self.lock` and decremented after. `/api/state` returns it as `"queue"`.
  - **`Station.inspect_sample`.** Signature: `Station.inspect_sample(file: str, alpha: float | None) -> dict`. It reads `root/demo/boards/<file>` and its `<stem>.golden.jpg` if present, uses the sample's design, and returns `inspection_json(iid)` plus `"known": match_known(...)` and `"sample": file`. For an assembled sample with no boxes, `"known"` is `{"good": bool, "verdict_agrees": bool}`, where the verdict agrees when (verdict == PASS) == good.
  - **`InspectReq.sample: str | None = None`.** When set, `/api/inspect` calls `inspect_sample`. A 404 unless demo mode is on and the file is in `load_samples`.
  - **`GET /api/samples`.** Returns a list of `{file, design, kind, good, marked}`. 404 when demo is off.
  - **`GET /api/samples/{file}.jpg`.** A thumbnail with longest side 160 px. It must reject path separators: only names present in `load_samples` are served.
  - **`GET /api/image/{iid}.jpg`.** The stored, unmarked image of an inspection (`store.image_path(row["image_sha"])`); 404 if absent.
  - **`GET /api/stats`.** Returns `Store.stats(now - shift_hours*3600)` plus `"shift_hours"`.
  - **`/api/state` additions:** `"demo": bool`, `"queue": int`, `"disclosure": str` (the exact line from Global Constraints when demo is on, else null).

- [ ] **Step 1: Write the failing tests** (`tests/test_app_demo.py`).
  - Fixture: a module-scoped `taught_client` built the way `tests/test_app_teach.py:14-46` does it (teach `demo-board` from `photo()` captures, `teach_windows: 30, teach_good: 20`), plus `"demo": True`.
  - It writes `demo/manifest.json` with two samples: `b4.png`, the cut board, with the hand-checked box `[690, 30, 710, 110]` around the cut; and `b0.png`, good with no boxes. Use `.png` names in the manifest, matching the files.

```python
def test_samples_listed_only_in_demo_mode(taught_client, plain_client):
    assert [s["file"] for s in taught_client.get("/api/samples").json()] == ["b4.png", "b0.png"]
    assert plain_client.get("/api/samples").status_code == 404

def test_inspecting_a_sample_returns_its_known_answer(taught_client):
    r = taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"}).json()
    assert r["verdict"] == "FAIL" and r["known"] == {"marked": 1, "found": 1, "extra": 0}
    assert taught_client.get(f"/api/image/{r['id']}.jpg").headers["content-type"] == "image/jpeg"

def test_unknown_sample_and_path_tricks_are_404(taught_client):
    for f in ["nope.png", "..%2Fdesigns%2Fx.yaml"]:
        assert taught_client.post("/api/inspect", json={"design": "demo-board", "sample": f}).status_code == 404
        assert taught_client.get(f"/api/samples/{f}.jpg").status_code == 404

def test_stats_reflect_inspections(taught_client):
    st = taught_client.get("/api/stats").json()
    assert st["boards"] >= 1 and st["shift_hours"] == 8

def test_a_new_inspection_does_not_wait_for_the_previous_advice(taught_client, monkeypatch):
    st = taught_client.app.state.station
    gate = threading.Event()
    real_template = st.advisor._template
    def slow_advise(record, history=None, **kw):     # Llama "still writing" until the gate opens
        gate.wait(5)
        return real_template(record, [])
    monkeypatch.setattr(st.advisor, "advise", slow_advise)
    t0 = time.perf_counter()
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"})
    taught_client.post("/api/inspect", json={"design": "demo-board", "sample": "b4.png"})
    gate.set()
    assert time.perf_counter() - t0 < 4.0
```

  - `plain_client` is the same config with `demo: False`. Build it cheaply: no teaching is needed for the 404 checks.
- [ ] **Step 2: Run them; expect FAIL** (404 on `/api/samples`, missing `known`).
- [ ] **Step 3: Implement the Station methods and routes listed in Interfaces.**
- [ ] **Step 4: Run `tests/test_app_demo.py` and `tests/test_app_teach.py`; expect all PASS.**
- [ ] **Step 5: Commit:** `git commit -m "Demo mode: test boards with known answers, plain images, shift stats, queue count"`

### Task 4: Uploads, kept in memory only

**Files:**
- Modify: `pcbinspect/app.py`. Add `Station.inspect_upload`, `Station.uploads`, the routes, and an advice-worker branch.
- Test: `tests/test_app_demo.py`

**Interfaces:**
- Consumes: `decode_upload`, `UploadError` (Task 2); `Station.waiting` and the lock pattern (Task 3).
- Produces:
  - **`Station.uploads`.** An `OrderedDict[str, dict]` of the last 32 results, keyed by `secrets.token_urlsafe(16)`. Each value is a payload with the same keys the frontend reads from `inspection_json`: `verdict, defects[], advice[], latency, gate, check_reason, advice_pending`, plus `"upload": True`, `"token": str` and `"uncalibrated": True`. Each defect has `id, defect_key, cls, name, x1, y1, x2, y2, confidence, size_px, location, severity, ipc`. `size_mm` and `width_mm` are None.
  - **`Station.inspect_upload`.** Signature: `Station.inspect_upload(data: bytes, alpha: float | None) -> dict`.
    - It calls `self.inspector.inspect(bgr, cfg["upload_design"], None, alpha, log=False, golden=None, force_review="not calibrated for this board")`.
    - The golden must not be used: add a parameter `use_golden: bool = True` to `Inspector.inspect` (`pcbinspect/pipeline.py:337`). When False it skips `golden_for`.
    - It stores the encoded JPEG in the payload cache (not the store), and queues advice as `(token, record, design)`.
    - It raises HTTP 503 "uploads need the design <name>" if `upload_design` is not loaded.
  - **The advice worker.** It accepts a str job key. For a str key it writes the advice into `self.uploads[key]["advice"]` (same shape as a store advice row: `advice, backend, model, latency_ms, decode_tps, rejected_reason`), and it never calls `store.log_advice`.
  - **`POST /api/inspect/upload`** (multipart field `file`) returns the payload. `UploadError` maps to its status. 404 when demo is off.
  - **`GET /api/upload/{token}`** returns the payload, and **`GET /api/upload/{token}.jpg`** the image. An unknown token is 404.

- [ ] **Step 1: Write the failing tests.**
  - `test_upload_is_inspected_and_never_logged`: post `photo()` as JPEG.
    - Expect 200, `verdict == "REVIEW"`, `uncalibrated` is True, `"token"` present, and `size_mm` is None on every defect.
    - The `inspections` row count is unchanged, and the image store holds no new file.
  - `test_upload_rejects_non_images_and_oversize`: a text file gives 400; 10 MB + 1 gives 413; a GIF gives 400.
  - `test_unknown_upload_token_is_404`: `GET /api/upload/abc` gives 404.
  - `test_upload_advice_arrives_in_memory`: after an upload of the cut board, poll `GET /api/upload/{token}` for up to 30 s until `advice` is non-empty. The template backend in tests is fine. The store's `advice` table is still empty for it.
- [ ] **Step 2: Run them; expect FAIL** (404).
- [ ] **Step 3: Implement.**
  - The multipart parsing needs `python-multipart`. Add it to `pyproject.toml` dependencies and to the Pi setup requirements if they are separate; check with `grep -rn "fastapi" deploy/pi_setup.sh pyproject.toml`.
- [ ] **Step 4: Run `tests/test_app_demo.py`; expect all PASS. Then run the full suite; expect 312 + new, all passing.**
- [ ] **Step 5: Commit:** `git commit -m "Demo mode: photo upload inspected in memory, advice included, nothing logged"`

### Task 5: Nightly demo reset

**Files:**
- Modify: `pcbinspect/app.py` (`Station.__init__`)
- Test: `tests/test_app_demo.py`

**Interfaces:**
- Consumes: `seconds_until` (Task 2) and `Store.reset` (Task 1).
- Produces: in demo mode only, a daemon thread that sleeps `seconds_until(cfg["demo_reset_hour"], datetime.now())`, then calls `self.reset_demo()`. `reset_demo()` takes `self.lock`, calls `store.reset()`, clears `self.cache` and `self.uploads`, and logs one INFO line.

- [ ] **Step 1: Write the failing test.** `test_reset_demo_empties_the_log`: after one sample inspection, `st.reset_demo()`; then `GET /api/stats` gives `boards == 0`, and `/api/state` gives `recent == []`.
- [ ] **Step 2: Run it; expect FAIL.**
- [ ] **Step 3: Implement it.** The thread starts only when `cfg.get("demo")` is set.
- [ ] **Step 4: Run it; expect PASS.**
- [ ] **Step 5: Commit:** `git commit -m "Demo mode: shared demo log resets nightly"`

### Task 6: Frontend foundation: tokens, fonts, shell, wording map

**Files:**
- Add: `pcbinspect/static/fonts/Archivo-400.woff2`, `Archivo-600.woff2`, `Archivo-800.woff2`, `ArchivoNarrow-500.woff2`, `ArchivoNarrow-700.woff2`.
  - Fetch them from the Google Fonts GitHub repo (`github.com/google/fonts/tree/main/ofl/archivo`, `.../archivonarrow`). That source is official and OFL.
  - Convert the TTF to woff2 with `fonttools` (`pyftsubset --flavor=woff2 --layout-features='*' --unicodes='U+0000-00FF,U+2010-2027,U+20B9'`). Add a line to `NOTICE.md`.
- Rewrite: `pcbinspect/static/index.html`.
- Create: `pcbinspect/static/app.css`, `pcbinspect/static/labels.js`, `pcbinspect/static/app.js`.
- Remove: the old Atkinson and Barlow fonts, once nothing references them (`grep -rn "Atkinson\|Barlow" pcbinspect`).

**Interfaces:**
- Produces, for Tasks 7–9:
  - **CSS custom properties** in `app.css` `:root`: `--bench --paper --ink --soft --line --blue --blue-2 --blue-3 --nav --nav-on --pass --check --check-lamp --fail`, with the exact values in Global Constraints.
  - **Component classes:** `.card .card-h .card-b .kpi .pill.reject .pill.ok .btn .btn.primary .btn.pass .btn.fail .balloon .seg .lamp .titleblock .meta`.
  - **`labels.js` exports** (as `window.Labels`):
    - `plainReason(text: string) -> string` maps the station's severity or measurement notes to plain words. Exact pairs:
      - "default for X (no golden-board measurement)" → "size from the detector box";
      - "no mm measurement: no neighbouring conductor inside the measured window" → "no reference measurement here";
      - "X is a functional defect: rejected at every IPC class" → "rejected at every IPC class";
      - anything else passes through unchanged.
    - `modelName(file: string) -> string`: "llama3_2_3b_spinquant" → "Llama 3.2 3B", "llama3_2_1b_spinquant" → "Llama 3.2 1B", and anything else unchanged.
    - `VERDICT = {PASS: {word: "PASS", line: "Board passed"}, REVIEW: {word: "CHECK", line: "look at this board"}, FAIL: {word: "FAIL", line: "board failed"}}`.
  - **`app.js` core:**
    - `api(path, opts) -> Promise<json>` shows errors in a toast with the server's `detail`.
    - `show(screen)` switches the sidebar and the `<section data-screen>`.
    - `state` holds the `/api/state` reply and is refreshed every 3 s.
    - `renderKpis(stats)` draws the six tiles; a null median shows "–".
  - **Shell:**
    - The sidebar has Inspect, Log, Trends, and a "Setup" group with Teach a board, Eye test, Customer rules and Ask the log. Its foot shows status from `/api/state`: vision runtime, `modelName(llm)`, and camera or "Test boards".
    - The header shows the screen title, the source switch (shown only when `state.demo`), and design, miss rate and operator. The operator name stays in `localStorage` (try/catch), as now.
    - The disclosure line sits under the header when `state.disclosure`.
    - Below 900 px the sidebar becomes a top bar with a menu button.

- [ ] **Step 1: Fetch the fonts and verify** they are valid woff2 with `python -c "from fontTools.ttLib import TTFont; [TTFont(f) for f in ...]"`. Check the licence file is OFL.
- [ ] **Step 2: Write the shell, tokens and `labels.js`.** Keep every existing element id that `tests/` or `deploy/` rely on: `grep -rn "getElementById\|querySelector" tests deploy bench`, and preserve what they use.
- [ ] **Step 3: Check in the browser.** This is the UI check procedure, reused in Tasks 7–9.
  - Copy the repo's `pcbinspect/` over `C:/Users/sulai/pcb_work/bundle/pcb-station/pcbinspect/`. Do not touch the bundle's `config/designs`, which holds the taught parts maps.
  - Merge the new keys from Task 3 into the bundle's `config/app.yaml`.
  - Then:
    `docker rm -f pcb-ui; MSYS_NO_PATHCONV=1 docker run -d --name pcb-ui -p 8000:8000 -v "C:/Users/sulai/pcb_work/bundle/pcb-station:/b" -w /b pcb-export python -m pcbinspect.app --config config/app.yaml --set camera=replay --set demo=true --set host=0.0.0.0`
  - Wait until `curl -s http://127.0.0.1:8000/api/state` answers, then open http://127.0.0.1:8000.
  - Take screenshots at 1440×900 and 390×844. Check: no font falls back (DevTools `document.fonts.check("16px Archivo")` is true), the sidebar and KPI tiles match the spec, and nothing overlaps.
- [ ] **Step 4: Run `labels.js` checks in the page** (DevTools evaluate):
  - `Labels.plainReason("default for spur (no golden-board measurement); no mm measurement: no neighbouring conductor inside the measured window")` → `"size from the detector box; no reference measurement here"`;
  - `Labels.modelName("llama3_2_3b_spinquant")` → `"Llama 3.2 3B"`.
- [ ] **Step 5: Commit:** `git commit -m "Dashboard: new shell, tokens, bundled Archivo fonts, plain wording map"`

### Task 7: Inspect screen

**Files:**
- Modify: `pcbinspect/static/app.js`, `app.css`, `index.html` (section `data-screen="inspect"`).

**Interfaces:**
- Consumes:
  - from Task 3: `/api/samples`, `/api/samples/{f}.jpg`, `/api/inspect` with `sample`, `/api/image/{iid}.jpg`, `/api/stats`, `state.queue`;
  - from Task 4: `/api/inspect/upload` and `/api/upload/{token}`;
  - from Task 6: Labels and classes.
  - Existing: `/api/review`, `/api/feedback`, `/api/undo`, `/api/advice/{id}/rate/{r}`, `/api/overlay/{iid}.jpg` (the "Marked" view fallback).
- Produces:
  - **`groupAdvice(steps, defects) -> [{ids: string[], cls, action, cause, step, cause_id, action_id}]`.** It groups steps with the same `cls + cause_id + action_id`. Groups follow the order of the steps; ids are listed in balloon-number order.
  - **`balloonNumber(defect) -> int`:** the number in `defect_key` ("D7" → 7).
  - **`renderInspection(r)`**, which draws:
    - the verdict bar: lamp, `VERDICT[...]`, a meta line with N findings, board ID and the verdict time from `latency.total_to_verdict`, plus the Pass/Fail/Inspect next buttons;
    - the board card: an `<img>` from `/api/image/{id}.jpg`, or the upload jpg, with an `<svg viewBox="0 0 W H">` overlay of balloons at each box's top-right corner; view toggles Marked / Camera / Reference; and the title block;
    - the findings table: the first 6 rows, then "Show all N"; a class switch 2/1/3 that reads `ipc.class1..3`; hovering or focusing a row toggles the `.hot` class on the row and on its balloon, and the reverse;
    - "What to do": the summary, grouped lines, a "details" toggle showing `cause_id, action_id, step`, and a footer with `modelName`, time and "Useful? Yes / No";
    - the known-answer card when `r.known`, and the "not calibrated for this board · sizes in pixels" note when `r.uncalibrated`.
  - **Test-board strip:** thumbnails from `/api/samples`, with the selected one outlined; clicking one inspects it. "Upload a photo" opens a file input that accepts `image/jpeg,image/png`. With no samples, it shows "No test boards installed".
  - **Empty state:** in demo mode, "Try the station on a test board" with one line of explanation; otherwise "Place a board and press Inspect". The space bar and foot pedal still inspect, and the space bar is ignored while typing in an input.
  - **Waiting:** while a request is pending, the Inspect button reads "Inspecting…", or "Waiting for N ahead" when `state.queue > 1`. While `advice_pending`, the advice card polls every 2 s.

- [ ] **Step 1: Implement the screen.**
- [ ] **Step 2: Check `groupAdvice`** on the real B00001 result (11 findings; that board's saved JSON is in `docs/img`, or re-inspect `deeppcb` sample 1) by DevTools evaluate:
  - every defect id appears in exactly one group;
  - the group count equals the number of distinct `cls + cause_id + action_id` triples.
- [ ] **Step 3: Visual check** in the container at 1440 and 390 px, with screenshots reviewed for:
  - a CHECK board with 11 findings;
  - a PASS board;
  - an assembled VisA board;
  - an upload (use `demo/boards/mendeley_002.jpg` as the file).

  Check each item:
  - [ ] hovering row 1 highlights balloon 1;
  - [ ] "Show all 11" shows 11 rows;
  - [ ] class switch 1 turns spur rows to OK;
  - [ ] the known-answer numbers equal `/api/inspect` JSON `known`;
  - [ ] no element overlaps another;
  - [ ] the page height for the 11-finding board is ≤ 2 viewports at 1440×900.
- [ ] **Step 4: Commit:** `git commit -m "Dashboard: Inspect screen with balloons, findings table, grouped advice, test boards and upload"`

### Task 8: Log and Trends screens

**Files:**
- Modify: `pcbinspect/static/app.js`, `app.css`, `index.html`.

**Interfaces:**
- Consumes the existing `state.recent`, `/api/export.csv` and `/api/spc/{design}`, and `openLogged(id)` behaviour (reopen a board).
- Produces:
  - **Log:** a table of time, board, design, verdict pill, findings, operator decision and advice by. Filters: verdict (All/PASS/CHECK/FAIL), design, and a board-ID search, applied in the browser to the last 200. Clicking a row reopens it in Inspect. "Download all as CSV" stays.
    - Needs `GET /api/log?limit=200`, returning `store.recent(limit)` with `n_defects` and `final_verdict`. Add it in `app.py` with a test in `tests/test_app_demo.py`: `test_log_lists_recent_with_counts` (after one sample inspection, the first row has `n_defects >= 1`).
  - **Trends:** two cards. The Pareto bars use `--blue` with the cumulative % in `--soft`. The p-chart is an SVG with dashed 3-sigma limits and points outside them in `--fail`.
    - When there are too few boards: "Needs at least 20 inspected boards (N so far)".
    - A plain note under each chart.

- [ ] **Step 1: Write the failing test** `test_log_lists_recent_with_counts`; run it; expect FAIL (404).
- [ ] **Step 2: Implement `/api/log` and both screens.**
- [ ] **Step 3: Run the test; expect PASS. Visual check** at 1440 and 390 px after inspecting 25 sample boards:
  - the filters narrow the rows;
  - clicking a row reopens the board;
  - the p-chart draws its limits.
- [ ] **Step 4: Commit:** `git commit -m "Dashboard: Log with filters and Trends charts in the new style"`

### Task 9: Setup screens: Teach, Eye test, Customer rules, Ask the log

**Files:**
- Modify: `pcbinspect/static/app.js`, `app.css`, `index.html`.

**Interfaces:**
- Consumes the existing endpoints, unchanged:
  - teach: `/api/teach/*`, `/api/enrol`, `/api/learn/*`;
  - eye test: `/api/eyetest/*`;
  - rules: `/api/rules*`;
  - ask: `/api/ask`;
  - capture: `/api/capture/*`.
- Produces: each screen as numbered step cards where the flow is a sequence (Teach: 1 name and type, 2 capture good boards, 3 finish; Eye test: 1 print, 2 photograph sheets, 3 result), one primary button per step, and the same fields and copy meaning as now.
  - "Learn from your corrections" is a card on Teach.
  - Rules shows the compiled rule, its checks and the capability verdict as a table.
  - Ask shows the question box and the answer with the query it used.

- [ ] **Step 1: Implement the four screens.**
- [ ] **Step 2: Check the whole teach flow through the new UI in the container.** Teach a design named `ui-check` from three replay captures with `mm_per_px` 0.0208, wait for "done", then inspect a sample of it.
- [ ] **Step 3: Check the other three in the browser:**
  - Ask: "How many boards failed today?" gets an answer;
  - Rules: compiling "Reject any open circuit" shows a rule;
  - Eye test: the card PDF link downloads.

  Screenshots at 1440 and 390 px.
- [ ] **Step 4: Run the full pytest suite; expect all PASS.**
- [ ] **Step 5: Commit:** `git commit -m "Dashboard: Teach, Eye test, Customer rules and Ask in the new style"`

### Task 10: Demo image, local run and the Hugging Face Space

**Files:**
- Create: `deploy/Dockerfile.demo`, `deploy/DEMO.md`, `deploy/space/README.md` (the Space card front-matter: `sdk: docker`, `app_port: 7860`).

**Interfaces:**
- Consumes everything above.
- Produces:
  - **The image.**
    - Base: `python:3.12-slim`. It installs `executorch==1.6.0` and the station's requirements from `pyproject.toml`, and copies `pcbinspect config kb deploy assets`.
    - `RUN --mount=type=secret,id=HF_TOKEN` runs `hf download Sulaiman-shariff/arm-bharat-pcb-inspect --include "models/vision/*" "models/llm/llama3_2_3b_spinquant.pte" "models/llm/tokenizer.model" "models/gate/*" "models/anomaly/*" "pi_bundle/pcb-station/demo/*" "pi_bundle/pcb-station/config/designs/*"`, then moves them into place.
    - CMD: `python -m pcbinspect.app --config config/app.yaml --set demo=true --set camera=replay --set host=0.0.0.0 --set port=7860`.
  - **`deploy/DEMO.md`:**
    - local: `docker build --secret id=HF_TOKEN,env=HF_TOKEN -f deploy/Dockerfile.demo -t pcb-demo .`, then `docker run -p 7860:7860 pcb-demo`;
    - the Space link, and what visitors can do.

- [ ] **Step 1: Build locally.**
  - Export `HF_TOKEN` from `.env` without printing it: `export HF_TOKEN="$(grep '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\"\r')"`.
  - Then `docker build --secret id=HF_TOKEN,env=HF_TOKEN -f deploy/Dockerfile.demo -t pcb-demo .`. Expect it to succeed.
  - `docker history --no-trunc pcb-demo | grep -c hf_` must print 0.
- [ ] **Step 2: Run it locally:** `docker run -d -p 7860:7860 --name pcb-demo pcb-demo`. In the browser:
  - inspect a DeepPCB sample, a VisA sample and an upload;
  - advice arrives with "Llama 3.2 3B";
  - the disclosure line shows.
- [ ] **Step 3: Create the Space**, private: `Sulaiman-shariff/pcb-inspection-demo`, Docker SDK, free CPU.
  - Use the `huggingface_hub` API: `create_repo(..., repo_type="space", space_sdk="docker", private=True)`, then `add_space_secret(repo, "HF_TOKEN", token)`.
  - Upload `deploy/Dockerfile.demo` as `Dockerfile`, `deploy/space/README.md` as `README.md`, and the code folders.
  - Assert `repo_info(...).private` is True before uploading.
- [ ] **Step 4: Wait for the build** (poll `get_space_runtime` until `RUNNING`, or read the build log on failure). Open the Space URL in the browser while logged in, and inspect a sample and an upload. Note the advice time on the free CPU for the report.
- [ ] **Step 5: Commit:** `git commit -m "Demo: Docker image for local runs and the Hugging Face Space"`

### Task 11: Docs, screenshots, final checks

**Files:**
- Modify:
  - `docs/TECHNICAL_REPORT.md` section 8 (images `docs/img/dashboard_check.png`, `dashboard_pass.png`, `dashboard_log.png`, with the caption timings re-measured);
  - `docs/DEMO_VIDEO_SCRIPT.md`;
  - `README.md`: a "Try it" section with the local command, the Space link (it says private until submission), and what the demo shows.

- [ ] **Step 1: Retake the three screenshots** from the local container (1440×900, demo off, with `camera=replay`, so they show the station, not the demo strip). Update each caption with its measured times.
- [ ] **Step 2: Update the video script** to the new screens. Keep its structure; change only what is on screen.
- [ ] **Step 3: Final checks:**
  - the full pytest suite passes;
  - in the arm64 simulator, `python deploy/arm_check.py` still passes against the rebuilt bundle (`python deploy/make_bundle.py`, then `docker run --rm --platform linux/arm64 -v <bundle>:/b pcb-pi-sim python /b/deploy/arm_check.py`).
- [ ] **Step 4: Commit:** `git commit -m "Docs: new dashboard screenshots, demo instructions, video script"`
