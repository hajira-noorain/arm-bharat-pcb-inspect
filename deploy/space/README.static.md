---
title: PCB Inspection Station
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: Offline PCB defect inspection with Llama 3.2 advice on ExecuTorch
---

# PCB Inspection Station (demo)

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5. The station is built for a Raspberry Pi 5 and runs
fully offline. This page is a **recording**: the station's own dashboard, showing exactly what the station's
code produced on each held-out test board (verdict, findings, known answer, and Llama's own advice). It ran
on an x86 computer's CPU, not the Pi; the Pi's timings are in the technical report.

- **Pick a test board** (never used for training).
- The station gives PASS, FAIL or CHECK, marks every finding on the board with its size and IPC class, and
  shows the board's known answer from the dataset.
- **Llama 3.2 3B on ExecuTorch** then explains the likely cause and the next action, choosing only from a
  checked defect knowledge base.

To upload your own photo, run the live demo locally with Docker: `deploy/DEMO.md` in the repository.

Built with Llama.
