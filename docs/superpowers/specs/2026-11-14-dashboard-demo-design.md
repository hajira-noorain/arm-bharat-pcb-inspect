# Dashboard redesign and public demo: design

Date: 2026-11-14. Status: approved in conversation, spec awaiting review.

## Goal

Make the station's UI clean and professional, and let anyone try it quickly with test pictures:
- **locally**, with one command;
- **online**, through a link to a Hugging Face Space.

The judges (Arm Bharat PS5) and anyone reading the repo should be able to:
- open it;
- pick a test board or upload a photo;
- see the verdict, the findings and Llama's advice within a minute;
- compare the result with the board's known answer.

The real station, with a camera and an operator on the Pi, uses the same dashboard. Demo mode only adds a picker for the board source.

## Decisions taken

| Question | Decision |
|---|---|
| Where the demo runs | Both: one local `docker run`, and a Hugging Face Space (private until submission) |
| What a visitor can inspect | Held-out test boards (with their known answer), or an uploaded photo |
| Scope | Every screen. Inspect gets the deepest rework |
| Look | Fresh design. The light inspection bench (A) combined with fabrication-drawing language (B), laid out as a professional dashboard |
| Llama online | 3B on the free CPU Space, clearly labelled as a shared cloud CPU, not the Pi |

The agreed mockup is `.superpowers/brainstorm/41799-1794628133/content/dashboard-v1.html`. It is not in git; it is a local reference only.

## Visual system

**Colours**
- Bench background `#EEF1F0`; panels white `#FFFFFF`; ink `#16201C`; secondary text `#5E6A66`; panel lines `#DCE3EE`.
- Drawing blue `#1D3F8F` for interactive elements, balloons, the selected state and title-block labels. Its tints are `#9DB0DA` and `#EEF2FB`.
- Sidebar navy `#14213F`, with the selected item `#25386A`.
- Verdict colours, as on the stack light: PASS `#2E7D32`, CHECK `#C98A0B` (lamp `#F2B12B`), FAIL `#C62828`.
- These colours are used only for verdicts and reject/OK pills, never for decoration.

**Type**
- Archivo (400–800) for text and headings, and Archivo Narrow (500–700) for tables, figures and title-block cells.
- Both are bundled as woff2 files in `pcbinspect/static/fonts/` (SIL OFL). Nothing is loaded from the internet, because the Pi is offline.

**Components**
- **Sidebar navigation.** Inspect, Log and Trends; a "Setup" group with Teach a board, Eye test, Customer rules and Ask the log; live status at the foot (Vision runtime, Llama model, camera or source).
- **Page header.** The page title; the source switch (Camera / Test boards / Upload, shown only when demo mode or upload is on); design, miss rate and operator.
- **Shift figures.** Six tiles: boards this shift, passed, sent to a person, failed, median verdict time and median advice time. Each count also shows its percentage.
- **Verdict bar.** A three-lamp stack light with one lamp lit, the verdict in words ("CHECK · look at this board"), a one-line summary (number of findings, board ID, verdict time), and the decision buttons (Pass board, Fail board, Inspect next). A coloured left edge in the verdict colour.
- **Board card.**
  - It holds the board picture with numbered blue balloons drawn in the browser (SVG over the image), and view toggles (Marked / Camera / Reference board).
  - A title-block footer gives design, scale, miss rate, what it was calibrated on, and the known answer in demo mode.
  - The card's height is capped so the page doesn't run long.
- **Findings card.**
  - One table row per finding: balloon number, defect, size, where, the result for the selected IPC class (as a pill), confidence, and on hover "Confirm · False alarm".
  - An IPC class switch (2 by default, then 1 and 3).
  - The first 6 rows show, then "Show all N".
  - Hovering or tapping a row highlights its balloon, and the reverse.
- **"What to do" card.**
  - Llama's summary, then the advice grouped: defects that share the same cause and action become one line ("Shorts 1, 3, 5, 6 · quarantine and continuity-test each. Likely: debris pressed on before etching (handling)").
  - Knowledge-base IDs and the process step sit behind a "details" toggle.
  - The footer says who wrote it and how long it took, and asks "Useful? Yes / No".
- **Known-answer card** (demo test boards only): "found 7 of 7 marked · 4 extra calls".

**Wording**
- Internal notes reach the operator in plain words. Examples:
  - "default for spur (no golden-board measurement); no mm measurement: no neighbouring conductor inside the measured window" becomes "size from the detector box; no reference measurement here";
  - raw model file names become "Llama 3.2 3B".
- The text mapping lives in one place in the frontend.

**Phone**
- Below 900 px the sidebar collapses into a top menu, the shift figures wrap into 2 columns, and the board card sits above the findings.

## Screens

1. **Inspect.** The layout above.
   - **Empty state (no board yet):** in demo mode, a short "Try the station on a test board" introduction, the test-board gallery and the upload target. On a real station: "Place a board and press Inspect", with the space bar or foot pedal still working.
2. **Log.**
   - A table with verdict pills; filters for verdict, design and date; search by board ID.
   - Clicking a row reopens the board in Inspect. CSV export stays.
3. **Trends.**
   - Two cards: defects by type (Pareto), and boards not passed per 20 boards with 3-sigma limits.
   - Plain notes explain what each shows. When there is too little data, the card says how many more boards are needed.
4. **Teach a board, Eye test, Customer rules, Ask the log.**
   - The same flows and endpoints as now, rebuilt as numbered step cards (where the content really is a sequence), with one primary button per step and consistent form fields.
   - "Learn from your corrections" stays on the Teach page.

All existing actions remain reachable. Nothing is removed from the station's behaviour.

## Demo mode

The setting `demo: true` in `config/app.yaml` (or `--set demo=true`) turns it on.

- **Test boards**
  - `GET /api/samples` lists a curated set of about 28 held-out boards from `demo/manifest.json`, four per design, covering bare and assembled, good and defective. Each entry gives file, design, kind, good and the number of marked defects.
  - Thumbnails are served by `GET /api/samples/{file}.jpg`.
  - `POST /api/inspect` with `{"sample": "<file>"}` inspects that image with its design's reference board and thresholds, through the same pipeline as a camera capture.
- **Known answer**
  - The inspection response carries `known` for a test board: the dataset's boxes, how many of them were found, and how many calls were extra.
  - A marked defect counts as found when a finding's box overlaps it with IoU ≥ 0.1, or a finding's centre falls inside it. These are the same rules the evaluation uses, so they are documented once.
  - Assembled boards with no boxes report good/faulty against the verdict.
- **Upload**
  - `POST /api/inspect/upload` (multipart) accepts JPEG or PNG only, checked by decoding the image, not by its name. The limit is 10 MB and 4000 px on a side.
  - It runs the bare-board detector only: there is no reference board, so there are no mm sizes, and sizes are shown in pixels with a note.
  - The image is kept in memory and is not written to the log or the image store.
  - The verdict on an upload is always CHECK, labelled "not calibrated for this board".
- **Shared online use**
  - Inspections run one at a time (a queue), and the page shows "waiting for 1 inspection ahead".
  - The demo database resets at 03:00 server time.
- **Disclosure line** in demo mode: "Demo on test boards. Online copy runs on a shared cloud CPU, not the Raspberry Pi; Pi timings are in the technical report."

## Statistics

`GET /api/stats?window=shift` returns:
- the counts of PASS, REVIEW and FAIL since the shift start (configurable; default 8 hours);
- the median verdict time and the median advice time;
- from the existing tables, the counts of boards, and the advice latencies.

It is used by the shift-figure tiles on both the real station and the demo.

## Running the demo

- **`deploy/Dockerfile.demo`**
  - x86 Python 3.12 with `executorch==1.6.0` and the station's requirements.
  - It copies the repo, and at build time downloads `models/*` and `pi_bundle/pcb-station/demo/*` from the private Hugging Face repo `Sulaiman-shariff/arm-bharat-pcb-inspect`, using the build secret `HF_TOKEN`.
  - It never bakes the token into a layer.
  - It starts `python -m pcbinspect.app --set demo=true --set camera=replay --set host=0.0.0.0 --set port=7860`.
- **Local:** `docker build --secret id=HF_TOKEN,env=HF_TOKEN -f deploy/Dockerfile.demo -t pcb-demo .`, then `docker run -p 7860:7860 pcb-demo`. `deploy/DEMO.md` documents both steps.
- **Online**
  - A Hugging Face Space (Docker SDK) named `Sulaiman-shariff/pcb-inspection-demo`, private, on free CPU hardware, built from the same Dockerfile, with `HF_TOKEN` as a Space secret.
  - Creating the Space publishes code to Hugging Face, which the user approved in conversation. It is made public only when the user says so, on submission day.
- **Llama online:** the 3B model, as on the station. The advice footer shows the real time it took.

## Testing

- **API tests (pytest)**
  - `/api/samples` lists only existing files.
  - Inspecting a sample returns `known` with the correct counts on a fixture with hand-checked boxes.
  - An upload rejects a non-image, an over-size file and an over-large resolution. It accepts a valid JPEG, writes nothing to the store, and returns CHECK.
  - `/api/stats` counts and medians are checked on a seeded store.
  - The queue serialises two concurrent inspections.
  - Demo mode off hides `/api/samples` and upload (404).
- **Frontend, checked by looking**
  - Every screen in the real local container, at 1440 px and 390 px wide, with screenshots reviewed.
  - Hovering a row highlights its balloon; "Show all" works.
  - Grouped advice matches the per-defect steps.
  - The known-answer card is right on two hand-checked boards.
  - Nothing overlaps (no floating button over content).
- **Unchanged:** all 312 existing tests pass, and `deploy/arm_check.py` still passes in the arm64 simulator.
- **Afterwards**
  - The dashboard screenshots in `docs/TECHNICAL_REPORT.md` section 8 are retaken.
  - `docs/DEMO_VIDEO_SCRIPT.md` is updated to the new screens.

## Out of scope

- The Llama fine-tuning dataset, which is designed separately after this.
- Choosing cause and action together in the advisor. This spec groups them as Llama gives them; making the action follow the chosen cause is a separate advisor change.
- Hindi, and user accounts.
