# AGENTS.md

Pre-implementation project: **no code has been written yet.** Do not assume any build, package, or test toolchain exists.

## Current state
- This directory is not a git repo and has no manifests, lockfiles, CI, or config of any kind.
- Only two files exist:
  - `draft-project-plan.md` — the sole source of truth. A technical spec + 16-week plan for an iOS system that predicts electronic bubble-craps dice outcomes (CV pose extraction → inverse-physics digital twin → Monte Carlo → EV/bet-flags).
  - `IMG_2111.MOV` — a single ~3.3 GB raw reference video capture (unannotated source footage, not tracked or labeled).

## Target architecture (from the plan, for when coding starts)
- Native iOS (Swift/UIKit), AVFoundation camera (120/240 fps), Metal compute C++/shaders, CoreML/ANE + YOLOv8-Keypoint, OpenCV for homography.
- Strict latency budget: end-to-end pipeline < 50 ms; Monte Carlo rollouts < 15 ms.

## Working conventions
- Any new source code or tooling is greenfield — there are no existing conventions, entrypoints, or file layouts to match.
- `draft-project-plan.md` is prose only (it contains pasted/duplicated Markdown artifacts); treat it as the design spec, not structured data.