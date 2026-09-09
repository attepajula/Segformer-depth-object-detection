# Data flow — segmentation service

Subscribes to the capture service's frame stream, finds doors and railings in
each frame with a neural network on the Mac GPU, and draws the result. Two
threads: one does the network, the other draws.

```mermaid
flowchart TB
    IN(["Capture service<br/>frame stream, ~30 fps"])

    subgraph WORK["Network thread"]
        direction TB
        RECV["Take the newest frame<br/>(older ones are skipped)"]
        UNPACK["Unpack it<br/>colour + depth images, camera lens info"]
        NET["Run the network<br/>SegFormer-B0 at 512×512 on the GPU<br/>→ a door/railing likelihood per pixel<br/>(~22 ms, the hard floor)"]
        MASK["Turn likelihood into a shape<br/>keep pixels the model is sure are 'door'<br/>(not just its best guess)"]
        DEPTH["Check it against depth<br/>merge blobs at the same distance into one door,<br/>reject anything wall-sized, tilted, or not flat"]
        HOLD["Require it to persist<br/>a door must show up 3 frames running,<br/>and it lingers 6 frames after it's lost<br/>— stops the flicker"]
        RECV --> UNPACK --> NET --> MASK --> DEPTH --> HOLD
    end

    HANDOFF{{"latest result<br/>handed across, newest-wins"}}

    subgraph DRAW["Display thread"]
        direction TB
        OVER["Draw the overlay<br/>tint the door orange, box + label,<br/>fps in the corner"]
        SHOW["Show the window / save a frame<br/>press q to quit"]
        OVER --> SHOW
    end

    IN ==> RECV
    HOLD --> HANDOFF --> OVER
    SHOW -.->|"while this draws frame N,<br/>the network thread is already on N+1"| RECV
```

## Notes

- **Why two threads.** The network call blocks its thread for ~22 ms waiting on
  the GPU. Drawing and unpacking (~15 ms) run on the other thread in that gap
  instead of adding to each frame. Effective rate ~26 fps; `--no-thread` runs it
  all in one thread if you want.
- **Why SegFormer, not the model in the name.** The job is *semantic
  segmentation* of fixed structure (doors, railings), not video object tracking.
  SegFormer is pretrained on a dataset that already has "door" and "railing".
- **The model is weak on its own** (small, general-purpose). Precision comes
  from the depth check and the persistence rule, not the network.
- **No shared code with capture.** This service has its own copy of the message
  format; a version mismatch is rejected, never misread.

Stage-by-stage detail and tuning knobs: [`../HANDOFF.md`](../HANDOFF.md).
