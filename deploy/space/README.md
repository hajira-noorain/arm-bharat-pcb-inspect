---
title: PCB Inspection Station
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Offline PCB defect inspection with Llama 3.2 advice on ExecuTorch
---

# PCB Inspection Station (demo)

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5. The station is built for a Raspberry Pi 5 and runs
fully offline; this Space runs the same code on a shared cloud CPU so anyone can try it.

- **Pick a test board** (never used for training) or **upload a photo** of a bare board.
- The station gives PASS, FAIL or CHECK, marks every finding on the board with its size and IPC class, and
  shows the board's known answer from the dataset.
- **Llama 3.2 3B on ExecuTorch** then explains the likely cause and the next action, choosing only from a
  checked defect knowledge base.

Times here are from a shared cloud CPU, not the Pi; the Pi's own timings are in the technical report. An
uploaded photo has no reference board, so it is judged by the detector alone, never passed automatically, and
its sizes are in pixels. Uploads are kept in memory only and never logged. The shared demo log is emptied
every night.

Built with Llama.
