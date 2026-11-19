import copy
import io
import pickle
import tarfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.benchmark import latency, measure, model_bytes
from src.compress import count_nonzero, make_permanent, prune_and_finetune, quantize_static, sparsity, to_half
from src.data import GPUData, _read_batches, synthetic
from src.models import ResNetCifar, count_params
from src.train import distill_loss, evaluate


def test_width_controls_size():
    assert count_params(ResNetCifar(64)) > 10_000_000
    assert count_params(ResNetCifar(32)) < count_params(ResNetCifar(64)) / 3.5
    assert ResNetCifar(8)(torch.randn(3, 3, 32, 32)).shape == (3, 10)


def test_training_learns_synthetic(trained, data):
    assert evaluate(trained, data) > 0.5  


def test_fusing_does_not_change_the_function(trained, data):
    x = data.xte[:32]
    fused = copy.deepcopy(trained).eval().fuse()
    assert (fused(x) - trained(x)).abs().max().item() < 1e-4


def test_int8_is_smaller_and_close(trained, data):
    q = quantize_static(trained, data.calibration(256))
    assert model_bytes(q)[0] < model_bytes(trained)[0] / 2.5
    assert evaluate(q, data, device="cpu") > evaluate(trained, data) - 0.15
    assert count_params(trained) == count_params(ResNetCifar(8))  


def test_half_precision_runs(trained, data):
    h = to_half(trained, torch.bfloat16)
    assert evaluate(h, data, dtype=torch.bfloat16) > 0.4


def test_pruning_hits_target_and_is_permanent(trained, data):
    pruned = prune_and_finetune(trained, data, 0.8, epochs=1, log=lambda s: None)
    assert sparsity(pruned) == pytest.approx(0.8, abs=0.01)
    assert count_nonzero(pruned) < 0.35 * count_nonzero(trained)
    assert not any(n.endswith("weight_orig") for n, _ in pruned.named_parameters())  
    assert sparsity(trained) < 0.01  
    assert model_bytes(pruned)[1] < model_bytes(trained)[1]  


def test_distill_loss_properties():
    s, y = torch.randn(16, 10), torch.randint(0, 10, (16,))
    assert distill_loss(s, torch.randn(16, 10), y, alpha=0.0).item() == pytest.approx(F.cross_entropy(s, y).item(), rel=1e-5)
    pure_soft = distill_loss(s, s.clone(), y, alpha=1.0)
    assert pure_soft.item() == pytest.approx(0.0, abs=1e-5)  
    assert distill_loss(s, torch.randn(16, 10), y, alpha=1.0).item() > 0


def test_augment_keeps_shape_and_is_reproducible():
    x = torch.randn(8, 3, 32, 32)
    a = GPUData.augment(x, torch.Generator().manual_seed(1))
    b = GPUData.augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape and torch.equal(a, b) and not torch.equal(a, x)


def test_cifar_parser_on_a_fake_archive(tmp_path):
    for name in ("data_batch_1", "test_batch"):
        with open(tmp_path / name, "wb") as f:
            pickle.dump({b"data": np.arange(5 * 3072, dtype=np.uint8).reshape(5, 3072), b"labels": [0, 1, 2, 3, 4]}, f)
    x, y = _read_batches(tmp_path, ["data_batch_1", "test_batch"])
    assert x.shape == (10, 3, 32, 32) and y.tolist() == [0, 1, 2, 3, 4] * 2


def test_benchmark_row_has_expected_fields(trained, data):
    row = measure("t", "precision", trained, data, devices=("cpu",), iters=3)
    assert row["cpu_ms_b1"] > 0 and row["cpu_img_per_s"] > 0 and 0 <= row["accuracy"] <= 1 and row["gzip_mb"] <= row["size_mb"] * 1.01
    assert latency(trained, "cpu", 4, iters=3) > 0
