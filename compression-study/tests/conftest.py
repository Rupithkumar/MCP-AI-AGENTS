import pytest
import torch

from src.data import GPUData, synthetic
from src.models import ResNetCifar
from src.train import train


@pytest.fixture(scope="session")
def data():
    return GPUData(synthetic(1500, 400), "cpu")


@pytest.fixture(scope="session")
def trained(data):
    torch.manual_seed(0)
    model = ResNetCifar(8)
    train(model, data, epochs=12, lr=0.1, batch=64, log=lambda s: None)
    return model.eval()
