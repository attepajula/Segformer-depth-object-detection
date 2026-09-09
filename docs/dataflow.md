# Data flow — `segformer run` (door/railing segmentation service)

Process: `python -m segformer run --backend mlx --overlay` (native **arm64**,
`segformer/.venv`, system python3.14). Consumes the schema-v1 ZMQ stream from
`capture serve`. Entry: `segformer/cli.py::cmd_run`.

Two run modes:
- **threaded** (default for `--overlay` / `--save`) — `segformer/runner.py::run_threaded`
- **serial** (`--no-thread`, and the `print` mode) — the loop in `cmd_run`

```mermaid
flowchart TD
    UP(["capture serve<br/>ZMQ PUB tcp://127.0.0.1:5556<br/>~1.54 MB msgpack blob / frame @ ~30 fps"])

    subgraph WORKER["worker thread  'segformer-infer'  (runner.py::worker)"]
        SUB["FrameConsumer(endpoint)  — segformer/consumer.py<br/>zmq.SUB · CONFLATE=1 · SUBSCRIBE=b'' · RCVTIMEO=2000 · LINGER=0<br/>connect tcp://127.0.0.1:5556<br/>(socket built INSIDE this thread — zmq sockets are single-thread)"]
        RECV["consumer.recv() → self._sock.recv()  (blocks, latest msg only)<br/>zmq.Again → None → continue"]
        DEC["segformer/wire.py::decode_frame(buf)<br/>msgpack.unpackb → check schema_version == 1 (WireError else)<br/>_unpack_array: np.frombuffer(data, dtype).reshape(shape)  (+lz4 if flagged)<br/>→ Frame(schema_version, frame_id, timestamp_ns, rgb (480,640,3) u8,<br/>depth (480,640) u16 mm, Intrinsics, depth_scale, registered, device)"]
        RST["frame.frame_id <= prev_id  → segmenter.reset()<br/>(replay loop / camera reopen → tracker.reset())"]
        INF["result, _ = segmenter.infer(frame, None)   — MlxSegmenter, see below"]
        TICK["RateMeter.tick() → state['fps']  (infer rate, 60-frame window)"]
        PUT["_Latest.put((frame, result))   — 1-slot, overwrites, never blocks"]
        SUB --> RECV --> DEC --> RST --> INF --> TICK --> PUT --> RECV
    end

    UP ==>|"TCP"| SUB

    subgraph INFER["MlxSegmenter.infer()  — segformer/backends/mlx_backend.py"]
        CP["model.class_probs(frame.rgb, [14,38,95,58])   — segformer/model/segformer.py<br/>_preprocess: cv2.resize 640×480 → 512×512, /255, ImageNet norm → (1,512,512,3)<br/>self(pixel_values): MiTEncoder (4 stages) + DecodeHead (all-MLP)<br/>  segformer/model/{mit.py, head.py}; weights fp16, norms fp32<br/>softmax(low)[:, :, class_ids]  → mx.eval  (blocks thread ~22 ms on GPU)<br/>cv2.resize (128×128×4) → (480,640,4) float32 bilinear"]
        subgraph PERCLASS["per target class ti / ade_id"]
            THR["binm = prob[:,:,ti] > threshold<br/>door/screen-door (14,58): 0.85   railing/bannister: 0.45<br/>cv2.morphologyEx OPEN 3×3"]
            CC["cv2.connectedComponentsWithStats(binm)<br/>cand = components with area >= 150 px"]
            DOOR{"door-like class<br/>and use_depth?"}
            BP["backproject(frame.depth, intrinsics)  — segformer/geometry.py<br/>LAZY: computed once, only if a door candidate exists<br/>z = depth/1000; points (H,W,3) m; valid = depth>0"]
            CONS["geometry.consolidate_door(labels, cand, stats, depth)<br/>anchor = largest blob; merge blobs whose median depth is within<br/>DEPTH_MERGE_TOL=0.35 m of the anchor → one region<br/>morphologyEx CLOSE 9×9 to bridge them"]
            GATE["geometry.door_gate(merged, points, valid, intr)  — runs ONCE<br/>metric width = bbox_px·z_med/fx; reject if:<br/>width>2.3 (wall) · not in 0.4–1.9 m · height not 0.9–2.7 m ·<br/>plane inlier<0.45 · |normal_y|>0.55<br/>no usable depth → 2D fallback (frame-frac, aspect)"]
            RAIL["railing/bannister: per component, no depth gate"]
            INST["_instance(id, ade_id, mask, prob_map):<br/>Instance(instance_id, label, score=prob[mask].mean(),<br/>mask (H,W) bool, box_xyxy)   — segformer/segmenter.py"]

            THR --> CC --> DOOR
            DOOR -->|yes| BP --> CONS --> GATE -->|keep| INST
            DOOR -->|no| RAIL --> INST
        end
        TRK["Tracker.update(SegResult)  — segformer/tracking.py<br/>match detections↔tracks per label by mask IoU (bbox-disjoint pre-filter)<br/>iou_match=0.3 · hit → hits++, score EMA 0.5 · miss → misses++<br/>report tracks with hits>=3 and misses<=3 · drop at misses>6"]
        CP --> PERCLASS --> TRK
        TRK --> OUT2["SegResult(frame_id, timestamp_ns, instances=[Instance…])"]
    end

    INF -.calls.-> INFER

    subgraph MAIN["main thread  (runner.py::run_threaded)"]
        GET["_Latest.get(timeout=0.5)  → (frame, result) | None"]
        RENDER["segformer/overlay.py::render(frame, result, fps=state['fps'])<br/>frame.rgb.copy(); per-instance alpha-blend mask (0.45);<br/>cv2 box + '{class} {score}'; banner 'frame N  inst K  F fps'<br/>fixed colour per class: door=orange, railing=green, bannister=yellow"]
        SAVE["--save DIR → cv2.imwrite(DIR/<frame_id>.png, img[:,:,::-1])"]
        SHOW["--overlay → overlay.show(img): cv2.imshow (MAIN thread — macOS)<br/>+ cv2.waitKey(1); 'q' → stop.set()"]
        GET --> RENDER --> SAVE --> SHOW --> GET
    end

    PUT -->|"latest (frame,result)"| GET

    subgraph SHUTDOWN["shutdown"]
        Q["'q' in window OR KeyboardInterrupt OR max_frames<br/>→ stop.Event set → worker loop ends → t.join(2 s)<br/>→ segmenter.close() (drop MLX model) → cv2.destroyAllWindows()"]
    end
    MAIN -.-> SHUTDOWN

    OUT2 -.-> DOWN2(["downstream consumer of SegResult<br/>(navigation / logging — not built)"])
```

## Serial mode (`--no-thread`)

Same stages, one thread, fully sequential per frame:
`consumer.frames()` → `decode_frame` → `seg.infer` → `overlay.render` →
`imwrite` / `imshow`. Everything (decode ~4 ms, infer ~29–39 ms, overlay ~1–4 ms,
imshow ~10 ms) adds up per frame. Threaded mode overlaps the non-infer work with
the next frame's inference.

## Key facts

| thing | value / location |
|---|---|
| process arch | native arm64 (`segformer/.venv`, python3.14) |
| model | SegFormer MiT-B0, ADE20K-150, ported to MLX; `segformer models` registry, weights `weights/segformer-b0-ade/model.mlx.safetensors` |
| model input | 512×512, ImageNet-normalized; output logits at 128×128×150 |
| targets | ADE ids 14 door, 38 railing, 95 bannister, 58 screen-door (`model/config.py::ADE20K_TARGETS`) |
| precision stack | (1) prob>0.85 not argmax → (2) `consolidate_door` merges same-depth blobs → (3) `door_gate` once → (4) temporal `Tracker` |
| GPU | MLX / Metal; `mx.eval` + `np.array` block the calling thread ~22 ms — GPU work is offloaded, the thread is not free |
| threads | 2 — worker (recv+decode+infer), main (overlay+imshow). `imshow` must be main-thread on macOS. `segmenter.infer` only ever called from the worker. |
| handoff | `runner._Latest` — 1 slot, latest-wins, producer never blocks |
| perf | infer ~29 ms (no door) / ~39 ms (door); threaded effective ~26 fps on a session2 replay |
| flicker | `Tracker`: 3 hits to confirm, lingers 6 frames (`min_hits`, `max_misses`) |
| coupling to xtion | none — `wire.py` + `frame.py` are an independent copy of schema-v1, guarded by `WIRE_SCHEMA_VERSION` |
