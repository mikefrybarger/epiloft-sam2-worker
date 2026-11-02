# epiloft-sam2-worker

RunPod Serverless worker that gives the Epiloft photo editor pixel-accurate masks
(trees, rooflines, poles, sky) using **SAM 2.1 (hiera base+)**, plus a guided-filter
soft-edge pass that removes the halo SegFormer leaves around tree crowns.

**Licenses:** SAM 2 code and weights are Apache 2.0. OpenCV is Apache 2.0. No GPL/AGPL.

## How it fits together

```
Photo opens in editor
  -> server fn calls  action: "embed"   (async /run, cold start is fine here)
  -> worker PUTs the embedding (~8 MB) to a Supabase signed upload URL
Every click
  -> browser runs sam2_decoder.onnx with that embedding   (instant, no network)
On "Apply" (optional)
  -> server fn calls  action: "segment" or "refine"  for the soft hi-res matte
```

All outputs go to **signed upload URLs** the app creates in Supabase storage, so the
worker never holds Supabase keys. If `upload_url` is omitted on `segment` / `sky` /
`refine`, the matte comes back inline as base64 PNG instead.

## RunPod setup

1. RunPod > Serverless > New Endpoint > **GitHub repo** `mikefrybarger/epiloft-sam2-worker`, branch `main`, Dockerfile at root.
2. GPU: **16 GB tier** (L4 / A4000 class) is plenty for base+. 24 GB if you later switch to large.
3. Workers: **min 0**, max 3. Idle timeout 60 s. Execution timeout 120 s.
4. Copy the endpoint ID into the app's env next to the splat and facade endpoint IDs.
5. One-time: call `get_decoder` with a signed upload URL to put `sam2_decoder.onnx` in Supabase storage, then serve it to the browser from there.

## Coordinates

- `points` for `segment` / `sky` are **original image pixels** `[x, y]`, after EXIF rotation (the way the browser shows the photo).
- Browser decoder points are in **1024 x 1024 input space**: `x * 1024 / width`, `y * 1024 / height` (SAM 2 squashes, it does not letterbox).
- `polygons` come back **normalized 0 to 1** `[x, y]`, ready to scale to any display size.

## Actions

### `embed`
```json
{ "input": { "action": "embed", "image_url": "...", "upload_url": "<signed PUT url>" } }
```
Returns:
```json
{
  "orig_size": [h, w],
  "input_size": 1024,
  "embedding": {
    "dtype": "float16",
    "byte_length": 8388608,
    "tensors": [
      { "name": "image_embed",      "shape": [1,256,64,64],   "byte_offset": 0 },
      { "name": "high_res_feats_0", "shape": [1,32,256,256],  "byte_offset": 2097152 },
      { "name": "high_res_feats_1", "shape": [1,64,128,128],  "byte_offset": 6291456 }
    ]
  },
  "ms": 900
}
```
The uploaded file is the three tensors back to back, float16, C order. The browser
slices by `byte_offset`, converts float16 to Float32Array, and feeds the decoder.

### `segment`
```json
{ "input": {
  "action": "segment", "image_url": "...",
  "points": [[1830, 940], [2100, 1500]], "labels": [1, 0],
  "box": null, "refine": true, "upload_url": "<optional signed PUT url>"
} }
```
Returns `score`, `polygons`, `matte_size` `[h, w]`, and `matte_uploaded` or `matte_png_base64`.
The matte is 8-bit alpha (255 = object), long side capped at 4096 px.

### `sky`
```json
{ "input": { "action": "sky", "image_url": "...", "upload_url": "<optional>" } }
```
Auto-seeds sky points along the top of the frame. Pass `points` / `labels` to override
(e.g. seeds taken from the existing SegFormer coarse mask). Returns `sky_found: false`
when the top of the frame has no bright smooth region.

### `refine`
```json
{ "input": { "action": "refine", "image_url": "...", "mask_url": "<binary mask PNG>", "upload_url": "<optional>" } }
```
Soft-edges a mask made in the browser decoder. Use this when the user hits Apply.

### `get_decoder`
```json
{ "input": { "action": "get_decoder", "upload_url": "<signed PUT url>" } }
```

### Browser decoder I/O (`sam2_decoder.onnx`)
| Input | Shape | Notes |
|---|---|---|
| image_embed | [1,256,64,64] | from embed file |
| high_res_feats_0 | [1,32,256,256] | from embed file |
| high_res_feats_1 | [1,64,128,128] | from embed file |
| point_coords | [1,N,2] | 1024 space |
| point_labels | [1,N] | 1 include, 0 exclude |
| mask_input | [1,1,256,256] | last logits for refinement clicks, else zeros |
| has_mask_input | [1] | 1 or 0 |

Outputs `masks` [1,3,256,256] logits and `iou_predictions` [1,3]. Take the best IoU,
upscale to image size, threshold at 0.

## Errors
Failures return `{ "error": "Type: message" }` with HTTP 200 from RunPod, so the app
should check for `error` in the output.

## Tuning (env vars)
`REFINE_MAX_SIDE` (default 4096), `EMBED_CACHE_SIZE` (default 4 photos kept warm),
`SAM2_CONFIG` / `SAM2_CHECKPOINT` to swap to the large model.
