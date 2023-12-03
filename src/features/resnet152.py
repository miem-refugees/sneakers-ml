import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet152, ResNet152_Weights
from tqdm.notebook import tqdm


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

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

    return numpy_features, classes
