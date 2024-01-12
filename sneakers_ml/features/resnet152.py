import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm.auto import tqdm


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


def get_resnet152_feature(image: Image.Image) -> np.array:
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    model.fc = Identity()

    device = "cpu"
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    preprocess = weights.transforms()
    preprocessed = preprocess(image)

    with torch.inference_mode():
        x = preprocessed.to(device).unsqueeze(0)
        prediction = model(x)
    return prediction.cpu().numpy()[0]


def get_resnet152_features(folder: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
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
        for data in tqdm(dataloader):
            x = data[0].to(device)
            prediction = model(x)

            features.append(prediction.cpu())

    full_images_features = torch.cat(features, dim=0)
    numpy_features = full_images_features.numpy()
    classes = np.array(dataset.imgs)

    return numpy_features, classes, dataset.class_to_idx
