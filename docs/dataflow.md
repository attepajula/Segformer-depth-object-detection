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
        NET["Run the network<br/>SegFormer-B0 at 512×512, MLX or PyTorch<br/>→ a door/railing likelihood per pixel<br/>(~22 ms on this Mac's GPU; more on CPU)"]
        MASK["Turn likelihood into a shape<br/>keep pixels the model is sure are 'door'<br/>(not just its best guess)"]
        DEPTH["Check it against depth<br/>merge blobs at the same distance into one door,<br/>reject anything wall-sized, tilted, or not flat"]
        HOLD["Require it to persist<br/>a door must show up 3 frames running,<br/>and it lingers 6 frames after it's lost<br/>— stops the flicker"]
        RESULT["Detections for this frame<br/>per door/railing: stable id, class,<br/>confidence, pixel mask, box"]
        RECV --> UNPACK --> NET --> MASK --> DEPTH --> HOLD --> RESULT
    end

    HANDOFF{{"newest (frame + detections)<br/>one slot, overwritten each frame"}}

    subgraph DRAW["Display thread"]
        direction TB
        OVER["Draw the overlay<br/>tint the door orange, box + label,<br/>fps in the corner"]
        SHOW["Show the window / save a frame<br/>press q to quit"]
        OVER --> SHOW
    end

    NAV(["downstream consumer — NOT BUILT<br/>detections + depth + lens →<br/>door plane in metres, for navigation"])

    IN ==> RECV
    RESULT --> HANDOFF --> OVER
    HANDOFF -.-> NAV
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
- **The model is weak on its own** (small, general-purpose). Precision comes
  from the depth check and the persistence rule, not the network.
- **No shared code with the camera side.** `stream/` carries its own copy of
  the frame contract; a version mismatch is rejected, never misread.

Stage-by-stage detail and tuning knobs: [`../HANDOFF.md`](../HANDOFF.md).
