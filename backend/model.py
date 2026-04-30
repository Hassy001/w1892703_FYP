import os
import torch
import torch.nn as nn
import torchvision.models as models


def load_model(device):
    if not os.path.exists("model_mc.pth"):
        raise FileNotFoundError("model_mc.pth not found. Run train_mc.py first.")

    ckpt = torch.load("model_mc.pth", map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    return model, classes
