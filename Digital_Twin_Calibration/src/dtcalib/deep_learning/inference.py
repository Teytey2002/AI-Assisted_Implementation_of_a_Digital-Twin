import torch
from model import RCInverseCNN


def load_model(path):
    model = RCInverseCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(model, x):
    with torch.no_grad():
        return model(x.unsqueeze(0)).item()
