from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .common import ROOT

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _read_batches(folder: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for name in names:
        with open(folder / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def load_cifar10(root: Path = ROOT / "data") -> dict[str, np.ndarray]:
    root.mkdir(parents=True, exist_ok=True)
    folder = root / "cifar-10-batches-py"
    if not folder.exists():
        archive = root / "cifar-10-python.tar.gz"
        if not archive.exists():
            print(f"downloading CIFAR-10 (about 163 MB) to {archive} ...")
            try:
                urllib.request.urlretrieve(CIFAR_URL, archive)
            except Exception as exc:
                raise SystemExit(f"Could not download CIFAR-10 ({exc}). Download {CIFAR_URL} yourself and put it in {root}.")
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(root)
    xtr, ytr = _read_batches(folder, [f"data_batch_{i}" for i in range(1, 6)])
    xte, yte = _read_batches(folder, ["test_batch"])
    return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}


def synthetic(n_train: int = 800, n_test: int = 300, seed: int = 0, classes: int = 10) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:32, 0:32] / 32.0

    def make(n):
        y = rng.integers(0, classes, n)
        imgs = np.empty((n, 3, 32, 32), np.uint8)
        for i, c in enumerate(y):
            ang = np.pi * c / classes
            g = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * (xx * np.cos(ang) + yy * np.sin(ang)) + rng.uniform(0, 6.28))
            colour = np.array([(c % 3 == 0) + 0.3, (c % 3 == 1) + 0.3, (c % 3 == 2) + 0.3])
            img = g[None] * colour[:, None, None] + rng.normal(0, 0.08, (3, 32, 32))
            imgs[i] = np.clip(img * 200, 0, 255).astype(np.uint8)
        return imgs, y.astype(np.int64)

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}


class GPUData:
    def __init__(self, raw: dict[str, np.ndarray], device: str | torch.device, subset: int = 0):
        self.device = torch.device(device)
        mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        prep = lambda x: ((torch.from_numpy(x).to(self.device).float() / 255.0) - mean) / std  
        n = subset or len(raw["xtr"])
        self.xtr, self.ytr = prep(raw["xtr"][:n]), torch.from_numpy(raw["ytr"][:n]).to(self.device)
        self.xte, self.yte = prep(raw["xte"]), torch.from_numpy(raw["yte"]).to(self.device)

    @staticmethod
    def augment(x: torch.Tensor, gen: torch.Generator | None = None, pad: int = 4) -> torch.Tensor:
        b, c, h, w = x.shape
        dev = x.device
        padded = torch.nn.functional.pad(x, (pad,) * 4)
        oy = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=gen)
        ox = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=gen)
        iy = (oy[:, None] + torch.arange(h, device=dev))[:, None, :, None]
        ix = (ox[:, None] + torch.arange(w, device=dev))[:, None, None, :]
        out = padded[torch.arange(b, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None], iy, ix]
        flip = torch.rand(b, device=dev, generator=gen) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    def train_batches(self, batch: int, gen: torch.Generator | None = None):
        perm = torch.randperm(len(self.xtr), device=self.device, generator=gen)
        for i in range(0, len(perm) - batch + 1, batch):  
            idx = perm[i : i + batch]
            yield self.augment(self.xtr[idx], gen), self.ytr[idx]

    def test_batches(self, batch: int):
        for i in range(0, len(self.xte), batch):
            yield self.xte[i : i + batch], self.yte[i : i + batch]

    def calibration(self, n: int = 512) -> torch.Tensor:
        return self.xtr[:n]
