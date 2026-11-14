# Trying the station without a Pi

Demo mode runs the real station code on any x86 computer with Docker, on held-out test boards instead of a
camera. A visitor can:

- pick a test board (DeepPCB, PKU-Market-PCB, real photos from the Mendeley set, and assembled VisA boards),
  none of them used for training, and see the verdict, every finding marked on the board, its known answer from
  the dataset, and Llama 3.2 3B's advice;
- upload a photo of a bare board. With no reference board for it, it is judged by the detector alone, never
  passed automatically, and its sizes are in pixels. Uploads stay in memory and are never logged;
- open the Log, Trends and setup screens as on the station.

The page says plainly that times in demo mode come from this computer's CPU, not the Pi.

## Locally

The model files are in a private Hugging Face repository, so the build needs a read token for it
(`HF_TOKEN`). The token is a build secret: it is used for the download step only and is not stored in the
image.

```bash
export HF_TOKEN=hf_...            # a read token with access to Sulaiman-shariff/arm-bharat-pcb-inspect
docker build --secret id=HF_TOKEN,env=HF_TOKEN -f deploy/Dockerfile.demo -t pcb-demo .
docker run --rm -p 7860:7860 pcb-demo
```

Open http://localhost:7860. The first build downloads about 4 GB (the models and the test boards). Llama's
advice takes longer on a laptop than on the Pi's own runtime; the verdict never waits for it.

## Online (recorded)

https://pcb-inspection-demo.vercel.app is the same dashboard in recorded mode: a static site showing exactly what
the station's code produced on each test board, plus Llama's answers to example shift questions and its
readings of example customer requirements. Nothing is edited; which shift questions appear as examples is
chosen at packaging time. To refresh it:

```bash
python deploy/record_demo.py --station http://127.0.0.1:8000 --out <recording>      # station running in demo mode
python deploy/build_static_demo.py --recording <recording> --out <site> [--questions ...]
cd <site> && vercel deploy --prod
```

A live online copy with Llama (a Docker Space) needs paid hosting, which is why the online demo is recorded.
The live demo, including upload, is the Docker image above.

## Settings

In `config/app.yaml`: `demo` (off on a real station), `upload_design` (whose detector judges uploads),
`samples_per_design`, `shift_hours` (the dashboard's shift figures), `demo_reset_hour` and
`demo_advice_queue`.
