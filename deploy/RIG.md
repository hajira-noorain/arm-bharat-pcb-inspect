# Camera rig guide for the Pi session (20-21 Nov)

Goal: every photo of every board looks the same, so any difference we see is a real defect and not the room.

## Kit (about Rs 1,000)
- **Camera stand or arm** with a mark (tape or pencil line) for the camera height. Target: lens 11-12 cm above the board.
- **2 pieces of linear polariser film.** One goes over the light, one over the lens. Turn the lens piece until the shiny glare on solder and copper disappears. Mark that angle with a pen and always use it. (Glare gone = the two films are "crossed".)
- **White diffuser** over the ring light: a translucent cup or tracing paper. Soft light means no hard glints.
- **Matte grey card** (any plain matte grey or white card, no shine).
- **Corner stops**: two cardboard L-brackets taped down, so each board slides into exactly the same spot.
- Printed sheets: the ChArUco board (`python deploy/calibrate_camera.py board`) and the 20 mm marker sheet. Print both at 100%, no "fit to page". Check the 50 mm line with a real ruler.

## Rules for every photo
1. Camera height fixed at the mark (11-12 cm). Never move it between photos.
2. Light on, diffuser and polarisers in place, room lights and curtains the same as always.
3. **Focus once**: put a marker or a detailed board in place and run `cam.focus_sweep()`. It tries lens positions and locks the sharpest. The depth of field is only a few mm, so redo this if the height changes.
4. Exposure and white balance lock themselves after settling. Do not touch them again. If the light changes, restart the camera and redo step 3.
5. Keep the ArUco marker and a small grey patch in the frame every time.
6. Take **4-frame averages**: `cam.capture(n_average=4)`. This cuts sensor noise by half.
7. For the golden board and the eye-test card also save a RAW file: `cam.capture_raw_dng("name.dng")` (open the camera with `raw=True`).
8. Save `cam.settings()` next to each photo (focus, exposure, gain, colour). Without it we cannot explain later differences.
9. After each lighting change take a flat-field frame (below).

## What to photograph, in order
1. **Lens calibration.** Print the ChArUco board, stick it flat on stiff card. Take 15-20 photos at the working height, tilting the card a little differently each time (up to about 25 degrees), board fully visible. Then run
   `python deploy/calibrate_camera.py calibrate photos/ --out calibration.json`.
   Good result: error under 1 px. Larger means blurry or bent card: retake.
2. **Flat-field card.** Lay the grey card where boards sit, same light, same height. Take 8 frames. This fixes dark corners (vignetting). Do it again if the light moves.
3. **Eye-test card.** Golden sheet 0 first, then test sheets 1 to 11, one after another, each in the corner stops. RAW for sheet 0.
4. **Real boards.** The golden (known-good) board first, with RAW. Then the boards to test, each with its marker and grey patch in view.
5. **Benchmarks (on the Pi, from the bundle folder).** `.venv/bin/python bench/bench_device.py --out results/device_bench.json` (detector on ExecuTorch vs the Arm runtime, anomaly backbone, Llama tokens/s, end-to-end), then `.venv/bin/python bench/soak.py --minutes 30` with the active cooler on (temperature and throttling over time). Copy the `results/` folder back to the laptop.

## Quick check before you start shooting
Take one photo and ask `imaging.capture_quality(img)`. It says if the picture is blurry, too dark, too bright or clipped. After the first good golden photo, build a reference with `CaptureProfile.from_images([...])`. Later photos are checked against it, so a bumped stand or a changed lamp is caught straight away.
