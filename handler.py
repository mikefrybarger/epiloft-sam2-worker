"""
Epiloft SAM 2 worker (RunPod Serverless).

Actions (input.action):
  embed        Compute the SAM 2 image embedding once per photo. The browser-side
               ONNX decoder then turns every click into a mask instantly.
  segment      Server-side click -> mask (+ optional soft-edge refine). Fallback for
               devices that can't run the browser decoder, and for final hi-res masks.
  sky          Whole-image sky matte with soft edges.
  refine       Soft-edge an existing binary mask (e.g. one made in the browser).
  get_decoder  Upload the ONNX mask decoder (built into the image) so the app can host it.

See README.md for the exact request/response contract.
"""

import base64
import io
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import requests
import runpod
import torch
from PIL import Image, ImageOps

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CFG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
MODEL_CKPT = os.environ.get("SAM2_CHECKPOINT", "/models/sam2.1_hiera_base_plus.pt")
DECODER_ONNX = os.environ.get("SAM2_DECODER_ONNX", "/models/sam2_decoder.onnx")
REFINE_MAX_SIDE = int(os.environ.get("REFINE_MAX_SIDE", "4096"))
CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "4"))
INPUT_SIZE = 1024  # SAM 2 squashes every image to 1024x1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- model (cold start)
t0 = time.time()
_model = build_sam2(MODEL_CFG, MODEL_CKPT, device=DEVICE)
PREDICTOR = SAM2ImagePredictor(_model)
print(f"[sam2] loaded {MODEL_CKPT} on {DEVICE} in {time.time() - t0:.1f}s", flush=True)

# Warm workers keep the last few photos' features so repeat clicks skip the encoder.
_feature_cache: "OrderedDict[str, dict]" = OrderedDict()


# ---------------------------------------------------------------- helpers
def _autocast():
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def load_image(url: str) -> np.ndarray:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    img = ImageOps.exif_transpose(img)  # match what the browser displays
    return np.array(img.convert("RGB"))


def set_image(image_url: str) -> np.ndarray:
    """Load image + run encoder, using the warm cache when possible."""
    cached = _feature_cache.get(image_url)
    if cached is not None:
        _feature_cache.move_to_end(image_url)
        PREDICTOR._features = cached["features"]
        PREDICTOR._orig_hw = cached["orig_hw"]
        PREDICTOR._is_image_set = True
        return cached["image"]

    image = load_image(image_url)
    with torch.inference_mode(), _autocast():
        PREDICTOR.set_image(image)
    _feature_cache[image_url] = {
        "features": PREDICTOR._features,
        "orig_hw": PREDICTOR._orig_hw,
        "image": image,
    }
    while len(_feature_cache) > CACHE_SIZE:
        _feature_cache.popitem(last=False)
    return image


def put_bytes(upload_url: str, data: bytes, content_type: str) -> None:
    r = requests.put(upload_url, data=data, headers={"Content-Type": content_type}, timeout=120)
    r.raise_for_status()


def resize_max_side(arr: np.ndarray, max_side: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = arr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return arr
    return cv2.resize(arr, (round(w * s), round(h * s)), interpolation=interp)


def soft_refine(image: np.ndarray, mask: np.ndarray, max_side: int = REFINE_MAX_SIDE) -> np.ndarray:
    """
    Turn a hard mask into a soft alpha matte along its edges with a guided filter.
    Interior and exterior stay solid; only a thin band around the boundary is
    re-estimated from the photo, which is what removes halos on tree crowns.
    Returns uint8 matte (0-255) at the refine resolution.
    """
    guide = resize_max_side(image, max_side).astype(np.float32) / 255.0
    h, w = guide.shape[:2]
    hard = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    hard = (hard > 127).astype(np.float32)

    band_px = max(3, round(max(h, w) / 300))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    band = (cv2.dilate(hard, k) - cv2.erode(hard, k)) > 0

    radius = max(4, round(max(h, w) / 400))
    soft = cv2.ximgproc.guidedFilter(guide=guide, src=hard, radius=radius, eps=1e-4)
    matte = hard.copy()
    matte[band] = np.clip(soft[band], 0.0, 1.0)
    return (matte * 255).astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, min_area_frac: float = 0.0002) -> list:
    """Outer contours as polygons in normalized [0,1] image coordinates."""
    binary = (mask > 127).astype(np.uint8) if mask.dtype == np.uint8 else mask.astype(np.uint8)
    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_frac * h * w:
            continue
        approx = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
        polys.append([[round(float(x) / w, 5), round(float(y) / h, 5)] for [[x, y]] in approx])
    return polys


def encode_png(matte: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", matte)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def deliver_matte(matte: np.ndarray, upload_url: str | None) -> dict:
    png = encode_png(matte)
    out = {"matte_size": [int(matte.shape[0]), int(matte.shape[1])]}  # [h, w]
    if upload_url:
        put_bytes(upload_url, png, "image/png")
        out["matte_uploaded"] = True
    else:
        out["matte_png_base64"] = base64.b64encode(png).decode("ascii")
    return out


def predict(points, labels, box=None, multimask=True):
    pc = np.array(points, dtype=np.float32) if points else None
    pl = np.array(labels, dtype=np.int32) if points else None
    bx = np.array(box, dtype=np.float32) if box else None
    with torch.inference_mode(), _autocast():
        masks, scores, _ = PREDICTOR.predict(
            point_coords=pc, point_labels=pl, box=bx, multimask_output=multimask
        )
    best = int(np.argmax(scores))
    return masks[best] > 0, float(scores[best])


# ---------------------------------------------------------------- actions
def action_embed(inp: dict) -> dict:
    if not inp.get("upload_url"):
        raise ValueError("embed requires upload_url (embedding is ~8 MB)")
    set_image(inp["image_url"])
    f = PREDICTOR._features
    tensors = [
        ("image_embed", f["image_embed"]),
        ("high_res_feats_0", f["high_res_feats"][0]),
        ("high_res_feats_1", f["high_res_feats"][1]),
    ]
    blob = io.BytesIO()
    meta = []
    for name, t in tensors:
        arr = t.detach().float().cpu().numpy().astype(np.float16)
        meta.append({"name": name, "shape": list(arr.shape), "byte_offset": blob.tell()})
        blob.write(arr.tobytes(order="C"))
    data = blob.getvalue()
    put_bytes(inp["upload_url"], data, "application/octet-stream")
    h, w = PREDICTOR._orig_hw[0]
    return {
        "orig_size": [int(h), int(w)],
        "input_size": INPUT_SIZE,
        "embedding": {"dtype": "float16", "byte_length": len(data), "tensors": meta},
    }


def action_segment(inp: dict) -> dict:
    image = set_image(inp["image_url"])
    points = inp.get("points") or []
    labels = inp.get("labels") or [1] * len(points)
    if not points and not inp.get("box"):
        raise ValueError("segment needs points and/or box")
    mask, score = predict(points, labels, inp.get("box"), inp.get("multimask", True))

    if inp.get("refine", True):
        matte = soft_refine(image, mask)
    else:
        matte = resize_max_side((mask * 255).astype(np.uint8), REFINE_MAX_SIDE, cv2.INTER_NEAREST)
    out = {"score": score, "polygons": mask_to_polygons(matte)}
    out.update(deliver_matte(matte, inp.get("upload_url")))
    return out


def _auto_sky_points(image: np.ndarray, n: int = 8):
    """Seed positive points in bright, low-texture spots of the top band, one negative at bottom."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (256, round(256 * h / w)))
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3) ** 2
    texture = cv2.blur(texture, (9, 9))
    band = slice(0, max(4, sh // 6))
    pts = []
    for i in range(n):
        x0, x1 = i * sw // n, (i + 1) * sw // n
        region_tex = texture[band, x0:x1]
        region_lum = gray[band, x0:x1]
        score = region_lum - 0.05 * region_tex  # bright and smooth wins
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        if region_lum[y, x] > 90:  # skip dark tops (no sky in that column)
            pts.append([(x0 + x + 0.5) * w / sw, (y + 0.5) * h / sh])
    labels = [1] * len(pts)
    pts.append([w / 2, h * 0.95])
    labels.append(0)
    return pts, labels


def action_sky(inp: dict) -> dict:
    image = set_image(inp["image_url"])
    if inp.get("points"):
        points, labels = inp["points"], inp.get("labels") or [1] * len(inp["points"])
    else:
        points, labels = _auto_sky_points(image)
        if sum(labels) == 0:
            return {"score": 0.0, "sky_found": False, "polygons": []}
    mask, score = predict(points, labels, multimask=True)
    matte = soft_refine(image, mask)
    out = {"score": score, "sky_found": True, "polygons": mask_to_polygons(matte)}
    out.update(deliver_matte(matte, inp.get("upload_url")))
    return out


def action_refine(inp: dict) -> dict:
    image = load_image(inp["image_url"])
    r = requests.get(inp["mask_url"], timeout=60)
    r.raise_for_status()
    m = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
    m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    matte = soft_refine(image, m)
    out = {"polygons": mask_to_polygons(matte)}
    out.update(deliver_matte(matte, inp.get("upload_url")))
    return out


def action_get_decoder(inp: dict) -> dict:
    if not inp.get("upload_url"):
        raise ValueError("get_decoder requires upload_url")
    with open(DECODER_ONNX, "rb") as fh:
        data = fh.read()
    put_bytes(inp["upload_url"], data, "application/octet-stream")
    return {"byte_length": len(data), "model": os.path.basename(MODEL_CKPT)}


ACTIONS = {
    "embed": action_embed,
    "segment": action_segment,
    "sky": action_sky,
    "refine": action_refine,
    "get_decoder": action_get_decoder,
}


def handler(job):
    inp = job.get("input") or {}
    action = inp.get("action")
    if action not in ACTIONS:
        return {"error": f"unknown action {action!r}; expected one of {sorted(ACTIONS)}"}
    t = time.time()
    try:
        result = ACTIONS[action](inp)
    except Exception as e:  # surface a clean error to the app
        return {"error": f"{type(e).__name__}: {e}"}
    result["ms"] = round((time.time() - t) * 1000)
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
