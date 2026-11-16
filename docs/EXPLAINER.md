# kibo-splat explainer

Your refresher for explaining this project to a recruiter or interviewer. Plain English first,
key technical terms kept and defined. Kept current as the project is built.

## The two-minute pitch

I built a 3D model of the inside of the International Space Station's Japanese module, Kibō, from
the camera footage of NASA's Astrobee robot, using a technique called 3D Gaussian splatting. I also
tested a question the one published Kibō splat left open: are the camera positions NASA ships
with the dataset accurate enough to train it, or do you have to recompute them from the images?

*(Results go here once they exist. No numbers until they're measured.)*

## How the system works

```mermaid
flowchart TD
    A["Astrobee frames<br/>(iva_kibo_trans, 1,068 grayscale)"] --> P1
    A --> P2
    P1["Poses A: NASA ground truth<br/>(groundtruth.txt)"] --> C
    P2["Poses B: COLMAP<br/>(structure from motion)"] --> C
    C["Initialize Gaussians<br/>(one per 3D point)"] --> T

    subgraph T["Training loop"]
        D["Pick a frame"] --> E["Render with gsplat"]
        E --> F["Loss: render vs frame"]
        F --> G["Update every Gaussian"]
        G --> H{"Densify step?"}
        H -- "yes" --> I["Split / clone / prune"]
        H -- "no" --> D
        I --> D
    end

    T --> J["Evaluate on held-out frames<br/>(PSNR, SSIM)"]
    J --> K["Compare A vs B:<br/>image quality + pose differences"]
    J --> L["Export .ply → web viewer"]
```

1. **Frames in.** 1,068 grayscale images from Astrobee's navigation camera as it flies 11.5 m through Kibō.
2. **Camera poses.** For each frame: where was the camera, and which way was it pointing? Two sources:
   NASA's ground truth (A), or COLMAP recomputing them from the images (B). This is the experiment.
3. **Initialize.** Start with one fuzzy blob (a Gaussian) per known 3D point.
4. **Training loop.** Render the blobs from a frame's camera, measure how different the render is from
   the real frame, nudge every blob to reduce that difference, and repeat thousands of times.
   Every so often, add blobs where detail is missing and delete useless ones (densification).
5. **Evaluate.** Score renders against frames the model never trained on.
6. **Compare and ship.** A vs B on quality and pose differences; export the best splat to a web viewer.

### Warm-up: the 2D toy

Before any cameras, `src/kibo_splat/toy2d.py` fits flat 2D Gaussians to a single image. It is the
training loop with the 3D parts removed, so the core idea can be seen on its own:

```mermaid
flowchart LR
    A["init_gaussians<br/>(random blobs)"] --> B["render<br/>(alpha compositing)"]
    B --> C["L1 loss<br/>vs target image"]
    C --> D["Adam step<br/>(nudge every Gaussian)"]
    D --> B
    B --> E["psnr vs<br/>mean_colour_image"]
```

The renderer is plain PyTorch and evaluates every Gaussian at every pixel, so the maths is readable
in a few lines. That costs memory, which is why the 3D trainer will use gsplat's tiled CUDA rasterizer.

## Key concepts

- **Gaussian splatting**: represent a scene as millions of small, fuzzy, colored 3D blobs. Each has a
  position, shape, rotation, color and opacity. Drawing them is fast enough for real time.
- **Novel view synthesis**: rendering the scene from an angle no photo was taken from. The goal of splatting.
- **Camera pose**: a camera's position plus orientation. Training needs one per image.
- **Parallax / triangulation**: seeing the same point from different *positions* pins down its 3D location.
  A camera that only turns (a panorama) gives no depth information.
- **COLMAP (structure from motion)**: software that recovers camera poses and a sparse 3D point cloud
  by matching features across overlapping images.
- **PSNR / SSIM**: standard image-similarity scores, measured on held-out frames.
- **Alpha compositing / transmittance**: how overlapping blobs combine into one pixel. Each blob adds
  its color × its opacity × the light that got through everything in front of it. That leftover light
  is the *transmittance*, T = Π(1 − α) over the blobs in front. The 2D toy uses this formula, and the
  3D trainer will use the same one; in 3D, "in front" means closer to the camera.
- **Reparameterization (sigmoid, exp)**: the optimizer works with unconstrained numbers, and we map
  them into their valid range when rendering. Opacity and color go through a sigmoid (always 0–1),
  and size goes through exp (always positive). Without that, a gradient step could produce a negative
  size or an opacity above 1.
- **L1 loss**: the average absolute pixel difference between render and target. Gradient descent
  (here the Adam optimizer) nudges every Gaussian to shrink it.
- **Null-case baseline**: a degenerate "model" a real result must beat. For the 2D toy, that's an
  image filled with the target's mean color. If our fit's PSNR isn't clearly above it, the fit has
  learned nothing.

## Challenges

- **Finding NASA imagery that splats.** Most NASA camera data turned out to be panoramas: the camera
  turns but barely moves. We measured this from metadata before downloading or training anything
  (e.g. the Node 2 sequence: 3 cm of movement while turning 145°). Full survey: `docs/data-candidates/`.

## Headline results

*(None yet.)*

## Honest limitations

- Grayscale, low-resolution navigation camera: no color, limited fine texture.
- The camera mostly looks along the module, so the splat is best from viewpoints near its path.
