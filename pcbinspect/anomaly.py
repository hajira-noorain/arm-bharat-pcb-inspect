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


def build_torch_backbone(arch: str = "resnet18"):
    """The exact module we export to ExecuTorch. Imported lazily so the Pi runtime needs no torchvision.

    arch: 'resnet18' (fast, 384-d patches) or 'wide_resnet50_2' (the PatchCore paper's backbone,
    1536-d patches, about 3-4x the compute).
    """
    import torch
    import torch.nn.functional as F
    import torchvision

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "resnet18":
                m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            elif arch == "wide_resnet50_2":
                m = torchvision.models.wide_resnet50_2(
                    weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(arch)
            self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
            self.layer2, self.layer3 = m.layer2, m.layer3
            self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

        def forward(self, x):
            f2 = self.layer2(self.stem(x))
            f3 = self.layer3(f2)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([self.pool(f2), self.pool(f3)], dim=1)

    return Backbone().eval()


def torch_feature_fn(device: str = "cpu", arch: str = "resnet18") -> FeatureFn:
    import torch
    net = build_torch_backbone(arch).to(device)

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


THUMB = 128


def grey_thumb(bgr: np.ndarray, size: int = THUMB) -> np.ndarray:
    """Small uint8 grey copy of an image; enough to derive the board foreground mask."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def foreground_mask(thumbs: list[np.ndarray], grid: tuple[int, int],
                    min_cov: float = 0.2, max_cov: float = 0.98) -> np.ndarray | None:
    """Board-foreground mask on the (gh, gw) feature grid from enrolment grey thumbnails.

    Cue: Otsu on the smoothed MEAN enrolment grey splits the frame into two intensity classes
    (board vs fixture/background). Which class is the board is decided by texture, not brightness:
    the class with the higher mean gradient magnitude (averaged over the individual images; a bare
    fixture is flat, a populated board is busy) is the foreground. The result is closed/opened to
    remove speckle, then reduced to the grid (a cell is foreground if >= half of it is).
    Returns None (mask disabled) if the image is flat or the mask covers < min_cov or > max_cov
    of the grid.
    """
    st = np.stack([np.asarray(t, np.float32) for t in thumbs])
    mean = cv2.GaussianBlur(st.mean(0), (0, 0), 2.0)
    if float(np.ptp(mean)) < 4.0:
        return None
    m8 = np.clip(mean, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(m8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.min() == binary.max():
        return None
    tex = np.zeros(mean.shape, np.float32)
    for t in st:
        gx, gy = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
        tex += np.sqrt(gx * gx + gy * gy)
    tex = cv2.GaussianBlur(tex / len(st), (0, 0), 2.0)
    fg_val = 1 if tex[binary == 1].mean() >= tex[binary == 0].mean() else 0
    m = (binary == fg_val).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    gh, gw = grid
    g = cv2.resize(m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) >= 0.5
    cov = float(g.mean())
    if cov < min_cov or cov > max_cov:
        return None
    return g


def _image_score(d: np.ndarray, mask: np.ndarray | None, topk_frac: float) -> float:
    """Image score from patch scores d (any shape): max, or mean of the top k = max(1, round(frac*n))
    patches; only patches inside the foreground mask (if any) count."""
    v = d.ravel() if mask is None else d.ravel()[mask.ravel()]
    if v.size == 0:
        return 0.0
    if topk_frac <= 0:
        return float(v.max())
    k = min(v.size, max(1, int(round(topk_frac * v.size))))
    return float(np.partition(v, v.size - k)[v.size - k:].mean())


def greedy_coreset(x: np.ndarray, k: int, seed: int = 0, proj_dim: int = 128) -> np.ndarray:
    """Greedy k-center coreset (PatchCore), computed in a random projection for speed. Returns indices."""
    n = len(x)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((x.shape[1], min(proj_dim, x.shape[1]))).astype(np.float32)
    z = x @ p
    first = int(rng.integers(n))
    if n * k > 2e7:   # large banks (offline evaluation): use a GPU if there is one; same algorithm
        try:
            import torch
            if torch.cuda.is_available():
                zt = torch.from_numpy(z).cuda()
                sel_t = [first]
                d = torch.linalg.norm(zt - zt[first], dim=1)
                for _ in range(k - 1):
                    i = int(torch.argmax(d))
                    sel_t.append(i)
                    d = torch.minimum(d, torch.linalg.norm(zt - zt[i], dim=1))
                return np.array(sel_t)
        except ImportError:
            pass
    sel = [first]
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
                 pixel_thr: float, design: str, version: int = 1, novelty_factor: float = 3.0,
                 input_size: int = INPUT, arch: str = "resnet18", *,
                 grid: tuple[int, int] | None = None, fg_mask: np.ndarray | None = None,
                 pos_mu: np.ndarray | None = None, pos_sd: np.ndarray | None = None,
                 topk_frac: float = 0.0, fg_status: str = "off"):
        self.feature_fn = feature_fn
        self.input_size = int(input_size)
        self.arch = arch
        self.bank = bank.astype(np.float32)
        self.nn = NearestNeighbour(self.bank)
        self.norm_scale = float(norm_scale)    # raw distance that maps to norm 0.5
        self.pixel_thr = float(pixel_thr)      # raw patch distance above which a patch is anomalous
        self.design = design
        self.version = version
        self.novelty_factor = novelty_factor
        self.history: list[dict] = []          # appended-feature log for undo
        # optional scoring refinements (all off by default = original behaviour)
        self.grid = tuple(grid) if grid is not None else None   # (gh, gw) the stats were built for
        self.fg_mask = None if fg_mask is None else np.asarray(fg_mask, bool)
        self.pos_mu = None if pos_mu is None else np.asarray(pos_mu, np.float32)
        self.pos_sd = None if pos_sd is None else np.asarray(pos_sd, np.float32)
        self.topk_frac = float(topk_frac)
        self.fg_status = fg_status

    # ---------------- scoring ----------------
    def normalise(self, raw: float | np.ndarray):
        return 1.0 - np.exp(-np.asarray(raw) * np.log(2) / self.norm_scale)

    def score(self, bgr: np.ndarray) -> AnomalyResult:
        t0 = time.perf_counter()
        x = preprocess(bgr, self.input_size)
        t1 = time.perf_counter()
        feat = self.feature_fn(x)
        t2 = time.perf_counter()
        _, c, gh, gw = feat.shape
        res = self.score_desc(descriptors(feat), gh, gw, bgr.shape[1], bgr.shape[0])
        res.latency_ms = {"anomaly_pre": (t1 - t0) * 1e3, "anomaly_backbone": (t2 - t1) * 1e3,
                          **res.latency_ms}
        return res

    def score_desc(self, desc: np.ndarray, gh: int, gw: int, img_w: int, img_h: int) -> AnomalyResult:
        """Score pre-computed patch descriptors (used by evaluation to avoid recomputing features)."""
        t2 = time.perf_counter()
        d = self.nn.dist(desc).reshape(gh, gw)
        if self.pos_mu is not None or self.fg_mask is not None:
            if self.grid is not None and (gh, gw) != tuple(self.grid):
                raise ValueError(f"feature grid {(gh, gw)} differs from the {self.grid} this model's "
                                 f"foreground mask / per-position statistics were built for "
                                 f"(input size {self.input_size}); use the same input size")
            if self.pos_mu is not None:
                d = (d - self.pos_mu) / self.pos_sd
            if self.fg_mask is not None:
                d = np.where(self.fg_mask, d, 0.0).astype(np.float32)
        t3 = time.perf_counter()
        regions = self._regions(d, img_w, img_h)
        raw = _image_score(d, self.fg_mask, self.topk_frac)
        return AnomalyResult(raw, float(self.normalise(raw)), d, regions, desc, {
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
            "input_size": self.input_size, "arch": self.arch, "history": self.history,
            "grid": list(self.grid) if self.grid is not None else None,
            "topk_frac": self.topk_frac, "fg_status": self.fg_status,
            "has_fg_mask": self.fg_mask is not None, "has_posnorm": self.pos_mu is not None}))
        extra = {}
        if self.fg_mask is not None:
            extra["fg_mask"] = self.fg_mask
        if self.pos_mu is not None:
            extra["pos_mu"], extra["pos_sd"] = self.pos_mu, self.pos_sd
        if extra:
            np.savez(d / "scoring_extras.npz", **extra)
        elif (d / "scoring_extras.npz").exists():
            (d / "scoring_extras.npz").unlink()

    @classmethod
    def load(cls, d: Path, feature_fn: FeatureFn) -> "AnomalyModel":
        d = Path(d)
        m = json.loads((d / "meta.json").read_text())
        obj = cls(feature_fn, np.load(d / "bank.npy"), m["norm_scale"], m["pixel_thr"], m["design"],
                  m["version"], m.get("novelty_factor", 3.0), m.get("input_size", INPUT),
                  m.get("arch", "resnet18"), grid=m.get("grid"), topk_frac=m.get("topk_frac", 0.0),
                  fg_status=m.get("fg_status", "off"))
        ex = d / "scoring_extras.npz"
        if ex.exists() and (m.get("has_fg_mask") or m.get("has_posnorm")):
            z = np.load(ex)
            if m.get("has_fg_mask") and "fg_mask" in z:
                obj.fg_mask = z["fg_mask"].astype(bool)
            if m.get("has_posnorm") and "pos_mu" in z:
                obj.pos_mu, obj.pos_sd = z["pos_mu"].astype(np.float32), z["pos_sd"].astype(np.float32)
        obj.history = m.get("history", [])
        return obj


def enrol(feature_fn: FeatureFn, good_images: list[np.ndarray], design: str,
          coreset_ratio: float = 0.1, max_bank: int = 16384, pixel_q: float = 99.9,
          seed: int = 0, input_size: int = INPUT, arch: str = "resnet18", *,
          fg_mask: bool = False, posnorm: bool = False, topk_frac: float = 0.0) -> AnomalyModel:
    """Build a model for a new design from good boards only (works with 5-10 images).

    Calibration uses leave-one-out: each good image is scored against a bank built WITHOUT it, so
    the normal-score statistics are honest even with very few images.
    Optional (keyword-only, default off): fg_mask, posnorm, topk_frac - see enrol_from_desc.
    """
    maps = [feature_fn(preprocess(im, input_size)) for im in good_images]
    feats = [descriptors(f) for f in maps]
    grid = (maps[0].shape[2], maps[0].shape[3])
    thumbs = [grey_thumb(im) for im in good_images] if fg_mask else None
    return enrol_from_desc(feature_fn, feats, design, coreset_ratio, max_bank, pixel_q, seed,
                           input_size, arch, fg_mask=fg_mask, posnorm=posnorm, topk_frac=topk_frac,
                           grid=grid, thumbs=thumbs)


def enrol_from_desc(feature_fn: FeatureFn, feats: list[np.ndarray], design: str,
                    coreset_ratio: float = 0.1, max_bank: int = 16384, pixel_q: float = 99.9,
                    seed: int = 0, input_size: int = INPUT, arch: str = "resnet18", *,
                    fg_mask: bool = False, posnorm: bool = False, topk_frac: float = 0.0,
                    grid: tuple[int, int] | None = None, thumbs: list[np.ndarray] | None = None,
                    pos_folds: int = 5, sd_floor_q: float = 0.1) -> AnomalyModel:
    """enrol() on pre-computed per-image descriptors.

    fg_mask   : ignore patches outside the board foreground (needs `thumbs`, grey thumbnails of the
                enrolment images from grey_thumb(); see foreground_mask for the cue and the 20%-98%
                coverage guard that disables it).
    posnorm   : patch score = (d - mu[pos]) / sd[pos], mu/sd of the 1-NN distance at each feature-grid
                position over the enrolment images. APPROXIMATION of leave-one-out: images are split
                into `pos_folds` folds (image i in fold i % K); each fold is scored against a bank
                built without it (exactly LOO when n <= K). sd is floored at the `sd_floor_q` quantile of
                all sd values. In this mode the same fold distances (standardised) also calibrate
                pixel_thr / norm_scale, so those live in z-score units. The shipped bank uses all images
                and is slightly denser than the fold banks, so z of real good boards is mildly negative.
    topk_frac : image score = mean of the top max(1, round(frac * n_patches)) patch scores (n_patches
                counts foreground patches only when a mask is active); 0 = max.
    grid      : (gh, gw) of the feature grid; inferred as a square from the patch count if omitted.
    """

    def build(fs):
        x = np.concatenate(fs)
        k = min(max_bank, max(1, int(len(x) * coreset_ratio)))
        return x[greedy_coreset(x, k, seed)]

    n_patch = len(feats[0])
    if (fg_mask or posnorm) and grid is None:
        side = int(round(n_patch ** 0.5))
        if side * side != n_patch:
            raise ValueError("grid=(gh, gw) is required for non-square feature grids")
        grid = (side, side)
    mask, fg_status = None, "off"
    if fg_mask:
        if not thumbs:
            raise ValueError("fg_mask needs the enrolment thumbnails (thumbs=[grey_thumb(img), ...])")
        mask = foreground_mask(thumbs, grid)
        fg_status = "on" if mask is not None else "disabled: flat image or coverage outside 20%-98%"
    mflat = None if mask is None else mask.ravel()

    pos_mu = pos_sd = None
    held_img_scores, held_patch = [], []
    if posnorm:
        n = len(feats)
        if n < 2:
            raise ValueError("posnorm needs at least 2 enrolment images")
        kf = min(pos_folds, n)
        D = [None] * n
        for f in range(kf):
            nn = NearestNeighbour(build([feats[j] for j in range(n) if j % kf != f]))
            for i in range(f, n, kf):
                D[i] = nn.dist(feats[i])
        D = np.stack(D)
        mu, sd = D.mean(0), D.std(0)
        sd = np.maximum(sd, max(float(np.quantile(sd, sd_floor_q)), 1e-6))
        pos_mu, pos_sd = mu.astype(np.float32).reshape(grid), sd.astype(np.float32).reshape(grid)
        for i in range(n):
            z = (D[i] - mu) / sd
            held_img_scores.append(_image_score(z, mflat, topk_frac))
            held_patch.append(z if mflat is None else z[mflat])
    elif 2 <= len(feats) <= 20:
        # few images: leave-one-out so every image is scored by a bank that never saw it
        for i in range(len(feats)):
            nn = NearestNeighbour(build([f for j, f in enumerate(feats) if j != i]))
            dist = nn.dist(feats[i])
            held_img_scores.append(_image_score(dist, mflat, topk_frac))
            held_patch.append(dist if mflat is None else dist[mflat])
    elif len(feats) > 20:
        # many images: calibrate on a held-out 20%, then build the shipped bank from all of them
        n_hold = max(4, len(feats) // 5)
        nn = NearestNeighbour(build(feats[n_hold:]))
        for f in feats[:n_hold]:
            dist = nn.dist(f)
            held_img_scores.append(_image_score(dist, mflat, topk_frac))
            held_patch.append(dist if mflat is None else dist[mflat])
    bank = build(feats)
    if held_patch:
        patch_d = np.concatenate(held_patch)
    else:  # a single image: no honest estimate possible, fall back to self-distances
        patch_d = NearestNeighbour(bank).dist(feats[0])
        held_img_scores = [_image_score(patch_d, mflat, topk_frac)]
        if mflat is not None:
            patch_d = patch_d[mflat]
    pixel_thr = float(np.percentile(patch_d, pixel_q))
    norm_scale = float(np.median(held_img_scores))
    if posnorm:
        norm_scale = max(norm_scale, 1e-3)
    return AnomalyModel(feature_fn, bank, norm_scale, pixel_thr, design, input_size=input_size, arch=arch,
                        grid=grid, fg_mask=mask, pos_mu=pos_mu, pos_sd=pos_sd, topk_frac=topk_frac,
                        fg_status=fg_status)
