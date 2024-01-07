import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm.autonotebook import tqdm


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_resnet152_features(folder: str, save=False):
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    model.fc = Identity()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preprocess = weights.transforms()
    dataset = ImageFolder(folder, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

    features = []
    with torch.inference_mode():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            prediction = model(x)

            features.append(prediction.cpu())

    full_images_features = torch.cat(features, dim=0)
    numpy_features = full_images_features.numpy()
    classes = np.array(dataset.imgs)

    if save:
        with open(Path("data", "features", "resnet152.pickle"), "wb") as f:
            pickle.dump((numpy_features, classes, dataset.class_to_idx), f)

    return numpy_features, classes, dataset.class_to_idx


if __name__ == "__main__":
    get_resnet152_features(str(Path("data", "merged", "images", "by-brands")), save=True)
