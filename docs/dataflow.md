# Data flow — segmentation service

Subscribes to an RGB-D frame stream — a live capture service, or a recorded
session replayed with `segformer replay` — finds doors, railings, and stairs
in each frame with a neural network on the GPU, then turns detections into
spatial audio. Two threads: one does the network, the other draws and plays
sound.

Everything about the stream (contract, codec, recording, replay, subscribe) is
the `stream/` package, separate from the net. The network runs through one of
two backends, same pipeline after it: **MLX** on Apple Silicon, or **PyTorch**
(CUDA / MPS / CPU) elsewhere. Pick with `--backend`, `$SEGFORMER_BACKEND`,
else `mlx`.

```mermaid
flowchart TB
    IN(["frame stream (ZMQ)<br/>live capture service, or<br/>segformer replay &lt;session&gt;"])

    subgraph WORK["Network thread"]
        direction TB
        RECV["Take the newest frame<br/>(older ones are skipped)"]
        UNPACK["Unpack it<br/>colour + depth images, camera lens info"]
        NET["Run the network<br/>SegFormer B0/B1/B2 at 512×512, MLX or PyTorch<br/>→ a door/railing/stairs likelihood, but only on a<br/>coarse 128×128 grid (the model downsamples 4×)<br/>(~38 ms mean on an M1 Max GPU, mlx b0; see bench table)"]
        UPSAMPLE["Stretch the likelihood back out<br/>128×128 → full frame size, smoothed<br/>(bilinear) — mask edges are only ever<br/>as sharp as that stretch, not pixel-exact"]
        MASK["Turn likelihood into a shape<br/>per-class threshold: 0.85 for door/screen-door<br/>(keep only what it's sure of), 0.45 for the rest"]
        DEPTH["Door / screen-door only: check against depth<br/>merge blobs at the same distance into one door,<br/>reject anything wall-sized, tilted, or not flat"]
        HOLD["Require it to persist<br/>3 frames running to confirm,<br/>lingers 6 frames after it's lost<br/>— stops the flicker"]
        RESULT["Detections for this frame<br/>per door/railing/stairs: stable id, class,<br/>confidence, pixel mask, box"]
        RECV --> UNPACK
        UNPACK ==>|"colour image"| NET
        NET --> UPSAMPLE --> MASK
        MASK -->|"door / screen-door"| DEPTH --> HOLD
        MASK -->|"railing / bannister / stairs<br/>(no depth gate)"| HOLD
        HOLD --> RESULT
        UNPACK -.->|"depth image + lens info<br/>(goes around the network, which only sees colour)"| DEPTH
    end

    HANDOFF{{"newest (frame + detections)<br/>one slot, overwritten each frame"}}

    subgraph DRAW["Display thread"]
        direction TB
        OVER["Draw the overlay<br/>tint the door orange, box + label,<br/>fps in the corner"]
        SHOW["Show the window / save a frame<br/>press q to quit"]
        OVER --> SHOW
    end

    subgraph NAV["consumer/ — on by default, --no-spatial to skip"]
        direction TB
        DIST["Distance to each object<br/>take the mask's pixels in the frame's depth,<br/>median of the valid readings<br/>no reading → unknown, not 0"]
        PLANE["Spatial-audio cue<br/>bearing + distance → pan + loudness<br/>smoothed per id, drawn as a radar"]
        SOUND["One tone per class<br/>door / railing / stairs, panned by direction<br/>1/2/3 mutes a class, a chirp confirms it<br/>(the user is blind — audio is the feedback)"]
        DIST --> PLANE --> SOUND
    end

    IN ==> RECV
    RESULT --> HANDOFF --> OVER
    UNPACK -.->|"the whole frame, depth included,<br/>untouched"| HANDOFF
    HANDOFF -.->|"frame (colour + depth + lens)<br/>+ detections"| DIST
    SHOW -.->|"while this draws frame N,<br/>the network thread is already on N+1"| RECV
```

## Notes

- **Why two threads.** The network call blocks its thread while the GPU works
  (mean 37.7 ms for b0/mlx on a Mac Studio M1 Max — see the bench table in
  `README.md`). Drawing and unpacking run on the other thread in that gap
  instead of adding to each frame. `--no-thread` runs it all in one thread
  if you want.

- **The overlay is a debug view, not the point of the project.** Every stage
  up to the network thread produces one thing: the per-frame detection list
  (`SegResult`: for each door/railing/stairs — a stable id, the ADE20K class,
  a confidence, a pixel mask, a box). The overlay just draws that for a
  sighted developer. `consumer/` is the real downstream — it takes that list
  plus the frame's depth and lens parameters, turns each detection into a
  direction and distance, and plays it as a tone panned toward that
  direction. That's the actual deliverable: a navigation aid for a blind
  user, not the overlay window.
- **Distance is not a segformer output.** The network sees the colour image
  only. Depth is used inside this service for one thing — the keep/reject check
  on each door — and no distance is kept from it. The frame, depth included,
  rides along untouched in the handoff; a consumer that needs distance takes
  the door mask, looks up those pixels in the frame's depth, and computes it
  itself. Nothing in the network thread produces or smooths a distance.
  `consumer/spatial.py` computes the cue (distance + bearing, smoothed per
  id) and draws it as a top-down radar; `consumer/audio.py` turns that cue
  into sound — one tone per class, panned by direction. Both run by default
  with `segformer run` (`--no-spatial` to skip, `--mute` to keep the radar
  without sound).
  What the consumer has to cope with:
  - *Sensor range.* The Xtion reads depth only from roughly 0.8 m to 3.5 m.
    Closer, or on a glass/reflective door, there is no reading — and the depth
    check here keeps such doors anyway, so a detection can arrive with no
    usable depth under its mask. Distance is then unknown, not zero.
  - *Mask edges.* The mask is a stretch of the 128×128 grid (note below), so
    its rim is blurry and bleeds onto the wall or floor behind the door. Use
    the median of the mask's interior, not every pixel, or the number gets
    dragged toward the background.
  - *Frame to frame.* A single frame's median is noisy. The detection's stable
    id is what lets the consumer follow one door across frames and smooth its
    distance per id.
- **The model is weak on its own** (small, general-purpose). For door /
  screen-door, precision comes from the depth check and the persistence
  rule, not the network. Railing, bannister, and stairs skip the depth
  check entirely — threshold + persistence is all they get today.
- **The network's own output is 128×128, not frame resolution.** SegFormer's
  decode head runs at input/4 (512/4) regardless of model size (b0/b1/b2 all
  share `strides=(4,2,2,2)`) — every downstream mask is a bilinear stretch of
  that coarse grid, so boundaries are never sharper than a ~4px-at-512 block,
  whatever the original camera resolution is.
- **No shared code with the camera side.** `stream/` carries its own copy of
  the frame contract; a version mismatch is rejected, never misread.

Stage-by-stage detail and tuning knobs: [`../HANDOFF.md`](../HANDOFF.md).
