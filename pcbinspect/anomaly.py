"""Patch-memory anomaly detection (PatchCore-style) with on-device learning by memory edits.

Backbone: frozen ResNet18 (ImageNet), layer2 + layer3 features, 3x3 local average, concatenated
-> one 384-d descriptor per 8x8-pixel patch of a 256x256 input. The backbone is the only neural
network and is exported to ExecuTorch; everything else is NumPy.

Learning during use is NOT backprop:
  enrol(design, good images)      -> new memory bank (coreset of good-patch descriptors)
  false_alarm(region)             -> region's descriptors appended to the bank (novelty-gated, undoable)
  calibration                     -> per-design score normalisation from held-out/leave-one-out good images
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

INPUT = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# FeatureFn: (N,3,INPUT,INPUT) float32 normalised -> (N, C, H, W) float32
FeatureFn = Callable[[np.ndarray], np.ndarray]


def preprocess(bgr: np.ndarray, size: int = INPUT) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def build_torch_backbone():
    """The exact module we export to ExecuTorch. Imported lazily so the Pi runtime needs no torchvision."""
    import torch
    import torch.nn.functional as F
    import torchvision

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
            self.layer2, self.layer3 = m.layer2, m.layer3
            self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

        def forward(self, x):
            f2 = self.layer2(self.stem(x))
            f3 = self.layer3(f2)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([self.pool(f2), self.pool(f3)], dim=1)

    return Backbone().eval()


def torch_feature_fn(device: str = "cpu") -> FeatureFn:
    import torch
    net = build_torch_backbone().to(device)

    def fn(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return net(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return fn


def executorch_feature_fn(pte_path: str | Path) -> FeatureFn:
    from executorch.runtime import Runtime
    import torch
    method = Runtime.get().load_program(str(pte_path)).load_method("forward")

    def fn(x: np.ndarray) -> np.ndarray:
        outs = []
        for i in range(x.shape[0]):  # the .pte is exported for batch 1
            outs.append(method.execute([torch.from_numpy(np.ascontiguousarray(x[i:i + 1]))])[0].numpy())
        return np.concatenate(outs)
    return fn


def descriptors(feat: np.ndarray) -> np.ndarray:
    """(N,C,H,W) -> (N*H*W, C)"""
    n, c, h, w = feat.shape
    return feat.transpose(0, 2, 3, 1).reshape(-1, c)


def greedy_coreset(x: np.ndarray, k: int, seed: int = 0, proj_dim: int = 128) -> np.ndarray:
    """Greedy k-center coreset (PatchCore), computed in a random projection for speed. Returns indices."""
    n = len(x)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((x.shape[1], min(proj_dim, x.shape[1]))).astype(np.float32)
    z = x @ p
    sel = [int(rng.integers(n))]
    d = np.linalg.norm(z - z[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
    return np.array(sel)


class NearestNeighbour:
    """Exact L2 nearest-neighbour distance to a memory bank (faiss if installed, else NumPy)."""

    def __init__(self, bank: np.ndarray):
        self.bank = np.ascontiguousarray(bank, dtype=np.float32)
        self._index = None
        try:
            import faiss
            self._index = faiss.IndexFlatL2(self.bank.shape[1])
            self._index.add(self.bank)
        except ImportError:
            self._sq = (self.bank ** 2).sum(1)

    def dist(self, q: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self._index is not None:
            d2, _ = self._index.search(q, 1)
            return np.sqrt(np.maximum(d2[:, 0], 0))
        out = np.empty(len(q), dtype=np.float32)
        for s in range(0, len(q), 2048):
            qq = q[s:s + 2048]
            d2 = (qq ** 2).sum(1)[:, None] + self._sq[None] - 2 * qq @ self.bank.T
            out[s:s + 2048] = np.sqrt(np.maximum(d2.min(1), 0))
        return out


@dataclass
class Region:
    bbox: tuple[int, int, int, int]   # full-image pixels
    score: float                      # max raw patch distance in region
    norm: float                       # normalised to [0,1]
    patch_idx: np.ndarray             # flat indices into the feature grid (for feedback)


@dataclass
class AnomalyResult:
    raw_score: float                  # image score = max patch distance
    norm_score: float
    heatmap: np.ndarray               # (H, W) raw distances on the feature grid
    regions: list[Region]
    grid_desc: np.ndarray             # (H*W, C) descriptors, kept for feedback
    latency_ms: dict[str, float] = field(default_factory=dict)


class AnomalyModel:
    def __init__(self, feature_fn: FeatureFn, bank: np.ndarray, norm_scale: float,
                 pixel_thr: float, design: str, version: int = 1, novelty_factor: float = 3.0):
        self.feature_fn = feature_fn
        self.bank = bank.astype(np.float32)
        self.nn = NearestNeighbour(self.bank)
        self.norm_scale = float(norm_scale)    # raw distance that maps to norm 0.5
        self.pixel_thr = float(pixel_thr)      # raw patch distance above which a patch is anomalous
        self.design = design
        self.version = version
        self.novelty_factor = novelty_factor
        self.history: list[dict] = []          # appended-feature log for undo

    # ---------------- scoring ----------------
    def normalise(self, raw: float | np.ndarray):
        return 1.0 - np.exp(-np.asarray(raw) * np.log(2) / self.norm_scale)

    def score(self, bgr: np.ndarray) -> AnomalyResult:
        t0 = time.perf_counter()
        x = preprocess(bgr)
        t1 = time.perf_counter()
        feat = self.feature_fn(x)
        t2 = time.perf_counter()
        _, c, gh, gw = feat.shape
        desc = descriptors(feat)
        d = self.nn.dist(desc).reshape(gh, gw)
        t3 = time.perf_counter()
        regions = self._regions(d, bgr.shape[1], bgr.shape[0])
        raw = float(d.max())
        return AnomalyResult(raw, float(self.normalise(raw)), d, regions, desc, {
            "anomaly_pre": (t1 - t0) * 1e3, "anomaly_backbone": (t2 - t1) * 1e3,
            "anomaly_knn": (t3 - t2) * 1e3, "anomaly_post": (time.perf_counter() - t3) * 1e3})

    def _regions(self, d: np.ndarray, img_w: int, img_h: int) -> list[Region]:
        gh, gw = d.shape
        mask = (d > self.pixel_thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sx, sy = img_w / gw, img_h / gh
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            idx = np.flatnonzero(lab.ravel() == i)
            peak = float(d.ravel()[idx].max())
            out.append(Region((int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)),
                              peak, float(self.normalise(peak)), idx))
        out.sort(key=lambda r: -r.score)
        return out

    # ---------------- learning by memory edits ----------------
    def false_alarm(self, result: AnomalyResult, region: Region, confirmed_twice: bool = False) -> dict:
        """Operator says the region is normal. Append its descriptors to the bank.

        Novelty gate: a region far beyond normal variation (peak > novelty_factor * pixel_thr) is
        likely a real defect mislabelled as fine; one wrong 'normal' label can blind the model to a
        whole defect class, so such regions need a second confirmation.
        """
        if region.score > self.novelty_factor * self.pixel_thr and not confirmed_twice:
            return {"accepted": False, "reason": "novelty_gate",
                    "message": "This area looks very different from good boards. Confirm again "
                               "(or ask a supervisor) before teaching it as normal."}
        new = result.grid_desc[region.patch_idx]
        start = len(self.bank)
        self.bank = np.concatenate([self.bank, new.astype(np.float32)])
        self.nn = NearestNeighbour(self.bank)
        self.version += 1
        entry = {"op": "append", "start": start, "count": len(new), "version": self.version}
        self.history.append(entry)
        return {"accepted": True, **entry}

    def undo_last(self) -> dict:
        if not self.history:
            raise ValueError("nothing to undo")
        e = self.history.pop()
        keep = np.ones(len(self.bank), bool)
        keep[e["start"]:e["start"] + e["count"]] = False
        self.bank = self.bank[keep]
        self.nn = NearestNeighbour(self.bank)
        self.version += 1
        return e

    # ---------------- persistence ----------------
    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "bank.npy", self.bank)
        (d / "meta.json").write_text(json.dumps({
            "design": self.design, "version": self.version, "norm_scale": self.norm_scale,
            "pixel_thr": self.pixel_thr, "novelty_factor": self.novelty_factor,
            "history": self.history}))

    @classmethod
    def load(cls, d: Path, feature_fn: FeatureFn) -> "AnomalyModel":
        d = Path(d)
        m = json.loads((d / "meta.json").read_text())
        obj = cls(feature_fn, np.load(d / "bank.npy"), m["norm_scale"], m["pixel_thr"], m["design"],
                  m["version"], m.get("novelty_factor", 3.0))
        obj.history = m.get("history", [])
        return obj


def enrol(feature_fn: FeatureFn, good_images: list[np.ndarray], design: str,
          coreset_ratio: float = 0.1, max_bank: int = 16384, pixel_q: float = 99.9,
          seed: int = 0) -> AnomalyModel:
    """Build a model for a new design from good boards only (works with 5-10 images).

    Calibration uses leave-one-out: each good image is scored against a bank built WITHOUT it, so
    the normal-score statistics are honest even with very few images.
    """
    feats = [descriptors(feature_fn(preprocess(im))) for im in good_images]

    def build(fs):
        x = np.concatenate(fs)
        k = min(max_bank, max(1, int(len(x) * coreset_ratio)))
        return x[greedy_coreset(x, k, seed)]

    held_img_scores, held_patch = [], []
    if 2 <= len(feats) <= 20:
        # few images: leave-one-out so every image is scored by a bank that never saw it
        for i in range(len(feats)):
            nn = NearestNeighbour(build([f for j, f in enumerate(feats) if j != i]))
            dist = nn.dist(feats[i])
            held_img_scores.append(float(dist.max()))
            held_patch.append(dist)
    elif len(feats) > 20:
        # many images: calibrate on a held-out 20%, then build the shipped bank from all of them
        n_hold = max(4, len(feats) // 5)
        nn = NearestNeighbour(build(feats[n_hold:]))
        for f in feats[:n_hold]:
            dist = nn.dist(f)
            held_img_scores.append(float(dist.max()))
            held_patch.append(dist)
    bank = build(feats)
    if held_patch:
        patch_d = np.concatenate(held_patch)
    else:  # a single image: no honest estimate possible, fall back to self-distances
        patch_d = NearestNeighbour(bank).dist(feats[0])
        held_img_scores = [float(patch_d.max())]
    pixel_thr = float(np.percentile(patch_d, pixel_q))
    norm_scale = float(np.median(held_img_scores))
    return AnomalyModel(feature_fn, bank, norm_scale, pixel_thr, design)
