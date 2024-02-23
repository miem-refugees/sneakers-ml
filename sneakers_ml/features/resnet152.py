import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from sneakers_ml.features.features import BaseFeatures


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet152Featues(BaseFeatures):
    def __init__(self, cfg_data: DictConfig, cfg_features: DictConfig, cfg: DictConfig) -> None:
        super().__init__(cfg_data, cfg_features)
        weights = ResNet152_Weights.DEFAULT

        self.preprocess = weights.transforms()
        self.device = self.get_device("cuda")

        if cfg.feature.onxx.use is True:
            pass
        else:
            self.model = resnet152(weights=weights)
            self.model.fc = Identity()
            self.model.to(self.device)
            self.model.eval()

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def get_feature(self, image: Image.Image) -> np.ndarray:
        preprocessed = self.apply_transforms(image)

        with torch.inference_mode():
            x = preprocessed.to(self.device).unsqueeze(0)
            prediction = self.model(x)
        return prediction.cpu().numpy()[0]  # type: ignore[no-any-return]

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
