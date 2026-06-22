---
name: customer-sdk
description: Customer-facing reference for the EyePop Python SDK. Covers the full workflow: dataEndpoint for dataset management and VLM ability registration, set_pop for configuring the inference pipeline, and the async worker for running inference on images, video files, and live streams. Use when building or explaining customer-facing EyePop integrations.
---

# EyePop Customer SDK Reference

**Docs:** https://docs.eyepop.ai/developer-documentation
**Pretrained models & abilities:** https://docs.eyepop.ai/developer-documentation/sdks/pretrained-models-and-abilities
**Abilities Hub** (published examples): https://www.eyepop.ai/abilities
**Dashboard** (build custom abilities): https://dashboard.eyepop.ai/dashboard

This skill covers the three building blocks every EyePop integration touches:

1. **`dataEndpoint`** — dataset management and custom VLM ability registration
2. **`set_pop`** — configure what the worker runs (built-in abilities, trained models, or custom VLM)
3. **Inference** — `upload` / `load_from` + `predict` loop

---

## Auth and environment setup

Always load credentials from a script-relative `.env`. Never hardcode keys.

```python
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent / ".env", override=True)
# override=True is mandatory — a stale shell var silently wins without it

API_KEY    = os.environ["EYEPOP_API_KEY"]   # eyp_... short key
ACCOUNT_UUID = os.environ["ACCOUNT_UUID"]   # required for dataEndpoint
EYEPOP_URL = os.getenv("EYEPOP_URL", "https://compute.eyepop.ai")
```

`.env` template:
```
EYEPOP_API_KEY=eyp_...
ACCOUNT_UUID=<uuid>
EYEPOP_URL=https://compute.eyepop.ai
```

**Key vs secret_key:**

| Parameter | Format | Use case |
|-----------|--------|----------|
| `api_key=` | `eyp_...` | Transient worker sessions on `compute.eyepop.ai` — **default for all demos** |
| `secret_key=` | Long encoded key | Named pop sessions (non-transient pop_id) |

Passing an `eyp_...` key as `secret_key=` raises `ValueError`. Always use `api_key=` for `eyp_...` keys.

If neither is passed, the SDK reads `EYEPOP_API_KEY` from the environment automatically after `load_dotenv`. This is the simplest form.

---

## Part 1 — Dataset SDK (`dataEndpoint`)

Use `dataEndpoint` to:
- Create and manage datasets
- Upload or import assets
- Set and read ground-truth annotations
- Register and publish custom VLM abilities

### Connection

```python
# Sync (simple scripts)
with EyePopSdk.dataEndpoint(api_key=API_KEY, account_id=ACCOUNT_UUID) as data:
    ...

# Async (preferred for bulk operations)
async with EyePopSdk.dataEndpoint(api_key=API_KEY, account_id=ACCOUNT_UUID, is_async=True) as data:
    ...
```

`eyepop_url` defaults to `EYEPOP_URL` env var.

### Create a dataset

```python
from eyepop.data.data_types import DatasetCreate

dataset = data.create_dataset(DatasetCreate(
    name="My Vehicle Dataset",
    description="Images for vehicle classification",
    tags=["vehicles", "demo"],
))
print(dataset.uuid)  # save this
```

### Upload an asset

```python
import time

with open("image.jpg", "rb") as f:
    job = data.upload_asset_job(f, mime_type="image/jpeg", dataset_uuid=dataset.uuid)
    asset = job.result()

# Wait for the platform to accept the asset before annotating
while True:
    asset = data.get_asset(asset.uuid, dataset_uuid=dataset.uuid)
    if asset.status == "accepted":
        break
    time.sleep(1)
```

### Import an asset from a URL

```python
from eyepop.data.data_types import AssetImport

job = data.import_asset_job(
    AssetImport(url="https://example.com/photo.jpg"),
    dataset_uuid=dataset.uuid,
    external_id="photo.jpg",   # optional — ties back to your record
)
asset = job.result()
```

### Set ground-truth annotation

```python
from eyepop.data.data_types import Prediction, PredictedClass

ground_truth = Prediction(
    source_width=1920,
    source_height=1080,
    classes=[PredictedClass(classLabel="forklift", confidence=1.0)],
)
data.update_asset_ground_truth(
    asset_uuid=asset.uuid,
    dataset_uuid=dataset.uuid,
    ground_truth=ground_truth,
)
```

### List assets with annotations

```python
assets = data.list_assets(dataset_uuid=dataset.uuid, include_annotations=True)
for asset in assets:
    print(asset.uuid, asset.status)
```

### Download an asset image

```python
stream = data.download_asset(asset.uuid)
with open(f"cache/{asset.uuid}.jpg", "wb") as f:
    f.write(stream.read())
```

---

## Part 2 — Register a custom VLM ability

Custom abilities let you define a VLM prompt once and reference it by alias in any Pop. Register once; the alias persists in the account.

> **Check first**: before registering, run `data.list_vlm_abilities()` to see if the ability already exists. Re-registering creates a duplicate group.

```python
# register_ability.py — run once
from eyepop import EyePopSdk
from eyepop.data.data_types import (
    VlmAbilityCreate,
    VlmAbilityGroupCreate,
    TransformInto,
    InferRuntimeConfig,
)

NAMESPACE   = "my-company"          # your EyePop namespace
ABILITY_NAME = f"{NAMESPACE}.image-classify.vehicle-type"
PROMPT = """Classify the vehicle in this crop as exactly one of:
- forklift: industrial lift truck with protruding forks
- vehicle: any other motorized vehicle

Respond with only the class label."""

with EyePopSdk.dataEndpoint(api_key=API_KEY, account_id=ACCOUNT_UUID) as data:
    # check for existing ability first
    existing = next((a for a in data.list_vlm_abilities() if a.name == ABILITY_NAME), None)
    if existing:
        print(f"Already exists: {existing.uuid}")
    else:
        group = data.create_vlm_ability_group(VlmAbilityGroupCreate(
            name=ABILITY_NAME,
            description="Classifies vehicle crops as forklift or vehicle",
            default_alias_name=ABILITY_NAME,
        ))
        ability = data.create_vlm_ability(
            create=VlmAbilityCreate(
                name=ABILITY_NAME,
                description="Classifies vehicle crops as forklift or vehicle",
                worker_release="qwen3-instruct",
                text_prompt=PROMPT,
                transform_into=TransformInto(classes=["forklift", "vehicle"]),
                config=InferRuntimeConfig(max_new_tokens=10, image_size=512),
                is_public=False,
            ),
            vlm_ability_group_uuid=group.uuid,
        )
        ability = data.publish_vlm_ability(ability.uuid, alias_name=ABILITY_NAME)
        ability = data.add_vlm_ability_alias(ability.uuid, alias_name=ABILITY_NAME, tag_name="latest")
        print(f"Registered: {ABILITY_NAME}:latest → {ability.uuid}")
```

Then reference it in your Pop:
```python
InferenceComponent(ability=f"{ABILITY_NAME}:latest")
```

**VLM ability parameters:**

| Field | Values | Notes |
|-------|--------|-------|
| `worker_release` | `"qwen3-instruct"` | Current default VLM — use this unless directed otherwise |
| `max_new_tokens` | `10`–`500` | Short for classification; longer for descriptions |
| `image_size` | `224`, `512`, `640`, `1024` | Higher = better detail, slower |
| `transform_into` | `TransformInto(classes=[...])` | Forces output to one of the named labels |
| `is_public` | `False` | Always `False` for customer abilities |

**Ability naming by task type:**

| Pattern | Result field | Use when |
|---------|-------------|----------|
| `<ns>.image-classify.<name>:latest` | `classes` | Single-label classification (one of N) |
| `<ns>.describe.<name>:latest` | `texts` | Free-text description or structured text output |

---

## Part 3 — `set_pop` and inference

### Define a Pop

```python
from eyepop.worker.worker_types import Pop, InferenceComponent

# Built-in ability
pop = Pop(components=[
    InferenceComponent(ability="eyepop.person:latest", confidenceThreshold=0.8)
])

# Custom VLM ability (registered with dataEndpoint above)
pop = Pop(components=[
    InferenceComponent(ability="my-company.image-classify.vehicle-type:latest")
])

# Custom trained model by UUID
pop = Pop(components=[
    InferenceComponent(abilityUuid="06a026006d9d74448000af297fa40074")
])
```

`abilityUuid` and `ability` are mutually exclusive — set one or the other. Never use the deprecated `modelUuid`.

### Async worker — images and video (preferred)

```python
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent

load_dotenv(Path(__file__).parent / ".env", override=True)
API_KEY = os.environ["EYEPOP_API_KEY"]

POP = Pop(components=[
    InferenceComponent(ability="eyepop.vehicle:latest", confidenceThreshold=0.5)
])

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


async def process(endpoint, image_path: Path) -> dict:
    cache = OUTPUT_DIR / f"{image_path.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    job = await endpoint.upload(str(image_path))
    result = await job.predict()
    cache.write_text(json.dumps(result, indent=2))
    return result


async def main():
    print("Connecting to EyePop...")
    async with EyePopSdk.async_worker(api_key=API_KEY) as endpoint:
        print("Connected.")
        await endpoint.set_pop(POP)
        for img in sorted(Path("input").glob("*.jpg")):
            result = await process(endpoint, img)
            objects = result.get("objects", [])
            print(f"{img.name}: {len(objects)} objects")

asyncio.run(main())
```

### Video file — frame-by-frame at 1 FPS

Use `target_fps=1` + a JSONL sidecar cache for VLM tasks on recorded video.

```python
async def process_video(endpoint, video: Path) -> list[dict]:
    cache = OUTPUT_DIR / f"{video.stem}.jsonl"
    if cache.exists():
        return [json.loads(l) for l in cache.read_text().splitlines()]
    results = []
    with open(cache, "w") as f:
        job = await endpoint.upload(str(video), target_fps=1)
        while result := await job.predict():
            results.append(result)
            f.write(json.dumps(result) + "\n")
    return results
```

### Live stream (RTSP / URL)

```python
async with EyePopSdk.async_worker(api_key=API_KEY) as endpoint:
    await endpoint.set_pop(POP)
    job = await endpoint.load_from("rtsp://camera-ip/stream")
    while result := await job.predict():
        # result is a live prediction frame
        print(result.get("objects", []))
```

### Sync worker (simple one-shot image scripts)

```python
with EyePopSdk.workerEndpoint(api_key=API_KEY) as endpoint:
    endpoint.set_pop(pop)
    result = endpoint.upload("image.jpg").predict()
```

### `set_pop` retry (VLM abilities need propagation time)

A freshly published VLM ability may not be available on the worker immediately. Retry with backoff:

```python
import time
from aiohttp import ClientResponseError

def set_pop_with_retry(endpoint, pop, attempts=6, first_wait=15):
    for i in range(attempts):
        try:
            endpoint.set_pop(pop)
            return
        except ClientResponseError as e:
            msg = str(e)
            if ("model uuids" in msg and "not found" in msg) or "couldn't resolve" in msg:
                if i == attempts - 1:
                    raise
                wait = first_wait * (2 ** i)
                print(f"  ability not ready yet, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
```

---

## Part 4 — Reading results

| Ability / task | Result key | Read with |
|----------------|-----------|-----------|
| Detection (`eyepop.*:latest`) | `objects` | `result.get("objects", [])` |
| Classification (`<ns>.image-classify.*`) | `classes` | `result.get("classes", [{}])[0].get("classLabel", "")` |
| Description (`<ns>.describe.*`) | `texts` | `result.get("texts", [{}])[0].get("text", "")` |
| OCR (nested) | `objects[*]["texts"]` | Walk nested objects |
| Custom trained (abilityUuid) | varies | Inspect a sample result to confirm shape |

Result envelope:
```json
{
  "objects": [{"classLabel": "person", "confidence": 0.95, "x": 100, "y": 50, "width": 80, "height": 200}],
  "classes": [{"classLabel": "forklift", "confidence": 0.91}],
  "texts":   [{"text": "DEFECTS: none observed."}],
  "seconds": 0.083,
  "source_width": 1920,
  "source_height": 1080
}
```

---

## Part 5 — Crop-then-classify (detect + refine)

Detect objects first, then classify each crop with a VLM — the standard two-pass pattern.

```python
from eyepop.worker.worker_types import Pop, InferenceComponent, CropForward

pop = Pop(components=[
    InferenceComponent(
        ability="eyepop.vehicle:latest",
        confidenceThreshold=0.5,
        forward=CropForward(targets=[
            InferenceComponent(ability="my-company.image-classify.vehicle-type:latest")
        ])
    )
])
```

Each detected vehicle crop is automatically forwarded to the VLM classifier. The crop-level result appears nested inside `objects[*].classes`.

---

## Full end-to-end example: register → set_pop → infer

```python
#!/usr/bin/env python3.12
"""
Step 1: run register_ability.py once.
Step 2: run this script to process images.
"""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent, CropForward

load_dotenv(Path(__file__).parent / ".env", override=True)
API_KEY   = os.environ["EYEPOP_API_KEY"]
NAMESPACE = os.environ["EYEPOP_NAMESPACE"]  # e.g. "my-company"

POP = Pop(components=[
    InferenceComponent(
        ability="eyepop.vehicle:latest",
        confidenceThreshold=0.5,
        forward=CropForward(targets=[
            InferenceComponent(ability=f"{NAMESPACE}.image-classify.vehicle-type:latest")
        ])
    )
])

INPUT_DIR  = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


async def main():
    print("Connecting...")
    async with EyePopSdk.async_worker(api_key=API_KEY) as endpoint:
        print("Connected.")
        await endpoint.set_pop(POP)
        for img in sorted(INPUT_DIR.glob("*.jpg")):
            job = await endpoint.upload(str(img))
            result = await job.predict()
            (OUTPUT_DIR / f"{img.stem}.json").write_text(json.dumps(result, indent=2))
            for obj in result.get("objects", []):
                label = (obj.get("classes") or [{}])[0].get("classLabel", "?")
                conf  = obj.get("confidence", 0)
                print(f"  {obj['classLabel']} → {label} ({conf:.2f})")

asyncio.run(main())
```

---

## Pretrained models and abilities

All abilities use the `:latest` tag unless you need a pinned version. Full list and composable pipeline docs: https://docs.eyepop.ai/developer-documentation/sdks/pretrained-models-and-abilities

For published example abilities built by EyePop, check the **Abilities Hub**: https://www.eyepop.ai/abilities — these are ready to use with `ability=` in your Pop without any registration step.

### Object detection

| Ability | Detects |
|---------|---------|
| `eyepop.common-objects:latest` | 80+ everyday objects (person, car, bottle, chair, laptop, …) |
| `eyepop.animal:latest` | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| `eyepop.vehicle:latest` | bicycle, car, motorcycle, bus, train, truck |
| `eyepop.vehicle.license-plate:latest` | License plate regions |
| `eyepop.device:latest` | clock, laptop, mouse, remote, keyboard, cell phone |
| `eyepop.sports:latest` | Frisbee, skis, snowboard, ball, kite, bat, glove, skateboard, surfboard, racket |
| `eyepop.localize-objects:latest` | Prompt-driven object localization |

### Person analysis

| Ability | What it returns |
|---------|----------------|
| `eyepop.person:latest` | Person bounding boxes |
| `eyepop.expression:latest` | Facial emotion: Happy, Neutral, Sad, Surprise, Angry, Fear, Disgust |
| `eyepop.person.pose:latest` | Human pose estimation |
| `eyepop.person.2d-body-points:latest` | 19 body keypoints (eyes, ears, shoulders, hips, elbows, wrists, knees, ankles, nose) |
| `eyepop.person.3d-body-points.full:latest` | Full 3D body pose |
| `eyepop.person.3d-body-points.heavy:latest` | Heavy 3D body pose model |
| `eyepop.person.3d-body-points.lite:latest` | Lightweight 3D body pose |
| `eyepop.person.3d-hand-points:latest` | 3D hand keypoints (25+ points) |
| `eyepop.person.face-mesh:latest` | Detailed facial geometry mesh |
| `eyepop.person.face.long-range:latest` | Face detection — distant subjects |
| `eyepop.person.face.short-range:latest` | Face detection — close range |
| `eyepop.person.palm:latest` | Palm region detection |
| `eyepop.person.reid:latest` | Person re-identification across frames |
| `eyepop.person.segment:latest` | Person silhouette segmentation |

### Text recognition (OCR)

| Ability | Description |
|---------|-------------|
| `eyepop.text:latest` | Text region detection / localization |
| `eyepop.text.recognize.landscape:latest` | OCR — landscape / signage text |
| `eyepop.text.recognize.landscape-tiny:latest` | Lightweight landscape OCR |
| `eyepop.text.recognize.square:latest` | OCR — document / square text |

### Other

| Ability | Description |
|---------|-------------|
| `PARSeq:latest` | Text recognition model |
| `EfficientSAM:latest` | Semantic segmentation |

> `eyepop.image-contents:latest` is deprecated — use a registered custom VLM `describe` ability instead.

---

## Decision guide

| Need | Choice |
|------|--------|
| Standard person / vehicle / face / object detection | Built-in `ability=` from the pretrained list above — no registration |
| Browse ready-to-use published abilities | Check https://www.eyepop.ai/abilities first |
| Classify crops into custom categories | Register VLM ability via `dataEndpoint`, use `CropForward` |
| Free-text description of images | Register VLM `describe` ability |
| Use a model trained in EyePop dashboard | `abilityUuid=<uuid from dashboard>` |
| Process images one by one | `endpoint.upload(path)` |
| Track objects across a video | `endpoint.upload(video_path)` — no `target_fps` |
| Extract frames for VLM | `endpoint.upload(video_path, target_fps=1)` |
| Live camera stream | `endpoint.load_from(rtsp_url)` |

---

## Related skills

| Skill | When |
|-------|------|
| `eyepop-agent-skills:python-sdk` | Full abilities table, typed Pop examples, result shapes, local mode |
| `eyepop-agent-skills:demo-patterns` | Script architecture decisions (stream vs file, single vs crop-then-classify) |
| `eyepop-agent-skills:model-discovery` | Check existing ability aliases before registering |
| `eyepop-agent-skills:auth` | Get API key / account UUID |

## Related wiki concepts

- [[async-worker-transient-session-is-canonical-python-sdk-entry-point]]
- [[custom-vlm-ability-registration]]
- [[custom-trained-model-referenced-by-ability-uuid]]
- [[crop-then-classify-two-pass-pattern]]
- [[jsonl-sidecar-cache-avoids-re-inference]]
- [[1fps-frame-extraction-for-video-upload]]
