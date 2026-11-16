# kibo-splat spec

Status: approved design, agreed 2026-11-16. Scope is locked in the README; this spec says *how* we
deliver it. Stage names match `docs/EXPLAINER.md`.

## Problem Statement

I want to understand 3D Gaussian splatting from the inside, and end with one result I can defend in
an interview. The one published splat of ISS Kibō threw away Astrobee's onboard camera poses as too
imprecise and recomputed them from the images. The public Astrobee dataset ships a *different*,
refined set of ground-truth poses, and nobody has tested whether those are good enough to train a
splat. I have three days.

## Solution

Train the same Kibō frames (`iva_kibo_trans`, 1,068 grayscale frames) twice:

- **Model A**: NASA's ground-truth poses.
- **Model B**: poses COLMAP recovers from the images.

Then compare them. **The headline is this experiment.** The flythrough demo comes with it. I learn
by writing my own trainer, proven first on Mip-NeRF 360 `garden`. If that trainer isn't good enough
by the end of Day 2, the A/B runs use gsplat's reference trainer instead, and the finding still stands.

**Scoreboard**

- **Primary: render quality.** PSNR on held-out frames (every 8th frame, the same frames for A and B).
- **Diagnostic: pose agreement.** After aligning COLMAP's poses to NASA's, how far apart they are
  (position error in cm, rotation error in degrees).

**Prediction, written before any run:** B ties or beats A. A tie means the two PSNRs are within
0.5 dB of each other. Reasoning: COLMAP's poses are fitted to these exact pixels; NASA's come from a
separate system, so any small mismatch, including timing, shows up as blur.

## User Stories

**Setup**

1. As the learner, I want the environment to pass preflight and the GPU smoke test on a CARC A40, so that I know the stack works before I spend time on it.
2. As the learner, I want every CARC job to charge the project account explicitly, so that no other group gets billed.
3. As the learner, I want the same code to run on my RTX 5070 laptop, so that I can iterate without a queue.
4. As the learner, I want both datasets fetched by one script, so that data never lives in git.

**Learning the technique**

5. As the learner, I want a 2D toy that fits Gaussians to a single image, so that I see the core loop (render, compare, nudge) before 3D adds cameras.
6. As the learner, I want my own 3D trainer to render `garden` recognizably, so that I know it works on a scene with published numbers to compare against.
7. As the learner, I want comments in the trainer that explain the *why* (e.g. why opacity goes through a sigmoid), so that the code teaches me as I read it later.

**Loading scenes**

8. As the learner, I want one loader that gives the same frame shape (image, intrinsics, pose, train/test label) for `garden`, Kibō with NASA's poses, and Kibō with COLMAP's poses, so that the trainer never knows which source it got.
9. As the learner, I want NASA's TUM-format poses converted to gsplat's world-to-camera convention by tested code, so that a flipped axis or a scalar-last quaternion can't silently ruin model A.
10. As the learner, I want each frame's NASA pose matched to its image by timestamp, with a check that every image found its pose, so that a timing offset can't hide.
11. As the learner, I want every 8th frame held out the same way for A and B, so that they're scored on identical frames.

**COLMAP**

12. As the learner, I want to run COLMAP alone on the Kibō frames on Day 1, using sequential matching, so that I know early whether model B is even possible.
13. As the learner, I want to know what share of frames COLMAP placed, so that I can apply the "at least half" rule.
14. As the learner, I want COLMAP to build model A's starting point cloud with NASA's poses locked, so that A and B both start from a triangulated cloud and differ only in their poses.

**Running the experiment**

15. As the learner, I want to approve each training run before it starts, with what runs, where and its rough cost, so that GPU spend stays under my control.
16. As the learner, I want A and B trained with identical settings, so that the poses are the only variable.
17. As the learner, I want one setup trained twice, if the budget allows, so that I know how much PSNR moves from noise alone and whether 0.5 dB is a fair tie band.
18. As the learner, I want each model to render its test frames from its *own* poses, so that the score reflects the whole pipeline from start to finish.
19. As the learner, I want A and B compared only on test frames that both have poses, so that COLMAP skipping a hard frame can't flatter B.

**Measuring**

20. As the learner, I want PSNR per test frame and the mean, so that I can find the frames where A and B differ most.
21. As the learner, I want COLMAP's poses aligned to NASA's with a similarity transform (scale, rotation, translation) before measuring error, so that COLMAP's arbitrary scale and orientation don't count as error.
22. As the learner, I want position error in cm and rotation error in degrees, so that the pose gap is in units a person understands.
23. As the learner, I want side-by-side renders of the worst frames, so that I can tell a *shifted but sharp* render (test-pose error) from a *blurry or ghosted* one (a worse 3D model).

**Shipping**

24. As the learner, I want a results table checked against the prediction, so that the outcome is honest even if I was wrong.
25. As the learner, I want the best splat exported and viewable in a web viewer, so that I have a demo to show.
26. As the learner, I want the explainer updated with the results and their caveats, so that I can explain the project to a recruiter.
27. As an interviewer, I want to see the prediction written before the results, so that I can trust the finding wasn't fitted after the fact.

## Implementation Decisions

**Modules**

- **Scene loading.** One module turns a dataset plus a pose source into a list of frames. Each frame holds a grayscale or colour image, a 3×3 intrinsics matrix, a 4×4 world-to-camera pose (gsplat's convention), and a train/test flag. There are three sources: `garden` (COLMAP's model ships with the dataset), Kibō + NASA ground truth (TUM format: `timestamp(ns) tx ty tz qx qy qz qw`, quaternion scalar-last, ISS world frame), and Kibō + COLMAP. Kibō intrinsics: 608.21 608.21 640 440, 1280×880, undistorted.
- **COLMAP runner.** Two modes: (1) a full reconstruction with sequential matching, for model B and for the Day 1 check; (2) triangulation only, with NASA's poses locked, for model A's starting cloud. It reports how many frames it placed.
- **2D toy.** Fits Gaussians to one image. A teaching step, kept separate from the 3D trainer.
- **Trainer.** Our own 3DGS training loop on gsplat's rasterizer: initialize from the point cloud, render, loss, update, densify (split / clone / prune). It only sees frames from scene loading. gsplat is always imported through the project's loader, which pins its build flags.
- **Pose comparison.** Aligns one trajectory to another with a Sim(3) fit, then reports position error (cm) and rotation error (degrees) per frame, plus the median and mean.
- **Evaluation.** Renders each test frame from that model's own pose and scores PSNR against the real frame. Also saves side-by-side images of the worst frames.
- **Export.** Writes the trained splat to `.ply` for a web viewer. Outputs stay out of git.

**Experiment rules**

- Held-out frames: every 8th frame by index, decided once and shared by A and B.
- A and B use identical trainer settings; only the pose source and its point cloud differ.
- COLMAP runs on all frames (test ones included), only to find poses. The splat never trains on test frames.
- If COLMAP places **at least half** the frames: compare on test frames both have. Report the share placed as part of the result.
- If it places **under half**: the finding is "COLMAP couldn't recover this sequence; NASA's poses were necessary". Train A for the demo; write up why B failed. No rescue attempts beyond trying the matching settings.
- Tie band: 0.5 dB. If a repeat run shows more noise than that, we report it and say the band was too tight; we don't quietly widen it.

**Plan**

| Day | Build | Done when |
|---|---|---|
| 1 | CARC setup, 2D toy, COLMAP-only run on Kibō | Smoke test passes on an A40; toy fits an image; we know COLMAP's placement rate |
| 2 | Own trainer on `garden`; Kibō scene loading with pose-conversion tests | Trainer renders `garden` recognizably; conversion tests pass |
| 3 | A/B runs, scoring, viewer, write-up | Results table with PSNR + pose error, checked against the prediction |

**Compute budget:** about 5 training runs (`garden` ×1–2, A, B, one repeat for noise), each approved
before it runs. A40 at about 8 SU/min plus CPU and memory. Run times are unknown until the smoke test
and the first `garden` run give a real number. The COLMAP check runs locally.

**Fallback:** if our trainer isn't good enough by the end of Day 2, A and B train on gsplat's
reference trainer. Its extra dependencies get added only then.

## Testing Decisions

- **A good test checks behaviour through the seam, against an independent truth.** Feed known
  inputs, check outputs we can work out by hand. No checks on internal steps, no mocking the thing
  under test. Every metric gets a null-case test: the degenerate input must score badly.
- **Seam 1: scene loading.** Tiny fixtures written by hand: a camera at a known position, facing a
  known direction, must project a known 3D point to a known pixel. That covers TUM's scalar-last
  quaternions, camera-to-world vs world-to-camera, and COLMAP vs gsplat conventions. Also: the split
  holds out exactly every 8th frame; a frame with no matching pose is reported, not silently dropped.
- **Seam 2: pose comparison.** Identical trajectories give 0. A scaled, rotated and shifted copy also
  gives ~0, which proves the alignment works. A constant (all-identical) trajectory or a shuffled one
  gives a large error. A known small perturbation gives back its known size.
- **Seam 3: scoring.** Identical images give infinite (or capped) PSNR. Known noise gives the
  expected PSNR. A render of the mean colour everywhere scores clearly below a real render.
- **Not unit-tested: the trainer.** It's GPU work; its check is end to end: `garden` renders
  recognizably and its PSNR is in a sensible range compared with published numbers (with their
  conditions noted).
- **Prior art:** the GPU smoke test checks against independent truths (a Gaussian straight ahead
  lands on the principal point; a gradient points the right way). The new tests follow that style.
- Tests run before any step is called done.

## Out of Scope

- Our own CUDA rasterizer (we use gsplat's).
- Colorization, not even as a stretch goal.
- Dynamic scenes, other ISS modules or sequences, apps built on top.
- Fine-tuning test poses before scoring (Option 3 in the design discussion). Built only if the renders show the shift-vs-blur question can't be settled by eye.
- Per-image appearance embeddings (the published Kibō splat needed them across five flights; we have one session).
- Rescuing a failed COLMAP run beyond trying the matching settings.

## Further Notes

- **Literature context, with its conditions:** the published Kibō splat (arXiv 2608.21685) reported
  31.2 dB PSNR from five April-2021 flights, ~5,000 images, GLOMAP poses and per-flight appearance
  embeddings. It found Astrobee's *onboard localizer* poses disagreed with SfM by 6.7 cm median. Our
  setup differs: one 2022 session, 1,068 frames, and the dataset's refined ground-truth poses, not
  the onboard ones. So that number is a hint, not a target.
- **Known data facts** (measured from `groundtruth.txt`): 1,068 poses over 229 s, 11.5 m camera path,
  3.8 m max baseline, 36° view-direction spread. The camera mostly looks *along* the module, which is
  close to forward motion: a known weak spot for COLMAP, hence the Day 1 check.
- **Risks:** (1) pose-convention bugs, covered by seam 1; (2) pose/image timing mismatch for model A,
  checked in scene loading; (3) PSNR punishing small test-pose shifts, which could inflate the gap,
  watched for with the worst-frame renders.
- Next step after approval: turn this spec into about 6–10 GitHub issues grouped by day, each with an
  acceptance checklist. CARC setup is issue #1.
