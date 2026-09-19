# Data flow — segmentation service

Subscribes to an RGB-D frame stream — a live capture service, or a recorded
session replayed with `segformer replay` — finds doors and railings in each
frame with a neural network on the GPU, and draws the result. Two threads: one
does the network, the other draws.

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
        NET["Run the network<br/>SegFormer-B0 (or b1/b2) at 512×512, MLX or PyTorch<br/>→ a door/railing likelihood, but only on a coarse<br/>128×128 grid (the model downsamples 4×)<br/>(~22 ms on this Mac's GPU; more on CPU)"]
        UPSAMPLE["Stretch the likelihood back out<br/>128×128 → full frame size, smoothed<br/>(bilinear) — mask edges are only ever<br/>as sharp as that stretch, not pixel-exact"]
        MASK["Turn likelihood into a shape<br/>keep pixels the model is sure are 'door'<br/>(not just its best guess)"]
        DEPTH["Check it against the depth image<br/>merge blobs at the same distance into one door,<br/>reject anything wall-sized, tilted, or not flat"]
        HOLD["Require it to persist<br/>a door must show up 3 frames running,<br/>and it lingers 6 frames after it's lost<br/>— stops the flicker"]
        RESULT["Detections for this frame<br/>per door/railing: stable id, class,<br/>confidence, pixel mask, box"]
        RECV --> UNPACK
        UNPACK ==>|"colour image"| NET
        NET --> UPSAMPLE --> MASK --> DEPTH --> HOLD --> RESULT
        UNPACK -.->|"depth image + lens info<br/>(goes around the network, which only sees colour)"| DEPTH
    end

    HANDOFF{{"newest (frame + detections)<br/>one slot, overwritten each frame"}}

    subgraph DRAW["Display thread"]
        direction TB
        OVER["Draw the overlay<br/>tint the door orange, box + label,<br/>fps in the corner"]
        SHOW["Show the window / save a frame<br/>press q to quit"]
        OVER --> SHOW
    end

    subgraph NAV["Downstream consumer — NOT BUILT"]
        direction TB
        DIST["Distance to each door<br/>take the door mask's pixels in the frame's depth,<br/>median of the valid readings<br/>no reading → unknown, not 0"]
        PLANE["Door in metres<br/>plane + aim point, from depth + lens"]
        DIST --> PLANE
    end

    IN ==> RECV
    RESULT --> HANDOFF --> OVER
    UNPACK -.->|"the whole frame, depth included,<br/>untouched"| HANDOFF
    HANDOFF -.->|"frame (colour + depth + lens)<br/>+ detections"| DIST
    SHOW -.->|"while this draws frame N,<br/>the network thread is already on N+1"| RECV
```

## Notes

- **Why two threads.** The network call blocks its thread while the GPU works
  (~22 ms on this Mac). Drawing and unpacking (~15 ms) run on the other thread
  in that gap instead of adding to each frame. Effective rate ~26 fps;
  `--no-thread` runs it all in one thread if you want.

- **The overlay is a debug view, not the output.** The product is the
  per-frame detection list (`SegResult`: for each door/railing — a stable id,
  the ADE20K class, a confidence, a pixel mask, a box). The overlay just draws
  it for a human. A real consumer (navigation, not built) would take that list
  plus the frame's depth and lens parameters and back-project the door into
  metres.
- **Distance is not a segformer output.** The network sees the colour image
  only. Depth is used inside this service for one thing — the keep/reject check
  on each door — and no distance is kept from it. The frame, depth included,
  rides along untouched in the handoff; a consumer that needs distance takes
  the door mask, looks up those pixels in the frame's depth, and computes it
  itself. Nothing in the network thread produces or smooths a distance.
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
- **The model is weak on its own** (small, general-purpose). Precision comes
  from the depth check and the persistence rule, not the network.
- **The network's own output is 128×128, not frame resolution.** SegFormer's
  decode head runs at input/4 (512/4) regardless of model size (b0/b1/b2 all
  share `strides=(4,2,2,2)`) — every downstream mask is a bilinear stretch of
  that coarse grid, so boundaries are never sharper than a ~4px-at-512 block,
  whatever the original camera resolution is.
- **No shared code with the camera side.** `stream/` carries its own copy of
  the frame contract; a version mismatch is rejected, never misread.

Stage-by-stage detail and tuning knobs: [`../HANDOFF.md`](../HANDOFF.md).
