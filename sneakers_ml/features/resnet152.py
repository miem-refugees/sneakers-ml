from collections.abc import Sequence

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from sneakers_ml.features.base import BaseFeatures
from sneakers_ml.models.onnx_utils import get_device, get_session, predict, save_torch_model


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet152Features(BaseFeatures):
    def __init__(self, config: DictConfig, config_data: DictConfig) -> None:
        super().__init__(config, config_data)

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.device = get_device(self.config.device)

        if self.config.use_onnx is True:
            self.onnx_session = get_session(self.config.onnx_path, self.device)
        else:
            self.model = self._initialize_torch_resnet()
            self.model.to(self.device)

    def _initialize_torch_resnet(self) -> torch.nn.Module:
        model = resnet152(weights=self.weights)
        model.fc = Identity()
        model.eval()
        return model  # type: ignore[no-any-return]

    def _create_onnx_model(self) -> None:
        model = self._initialize_torch_resnet()
        torch_input = torch.randn(1, 3, 224, 224)
        save_torch_model(model, torch_input, self.config.onnx_path)

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def _get_feature(self, image: Image.Image) -> np.ndarray:
        return self.get_features([image])

    def get_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])

        if self.config.use_onnx is True:
            return predict(self.onnx_session, preprocessed_images)

        with torch.inference_mode():
            x = preprocessed_images.to(self.device)
            prediction = self.model(x)
        return prediction.cpu().numpy()  # type: ignore[no-any-return]

    def get_features_folder(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        if self.config.use_onnx is True:
            raise NotImplementedError

        self._create_onnx_model()

        dataset = ImageFolder(folder_path, transform=self.preprocess)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

        features = []
        with torch.inference_mode():
            for data in tqdm(dataloader, desc=folder_path):
                x = data[0].to(self.device)
                prediction = self.model(x)

                features.append(prediction.cpu())

        full_images_features = torch.cat(features, dim=0)
        numpy_features = full_images_features.numpy()
        classes = np.array(dataset.imgs)

        return numpy_features, classes, dataset.class_to_idx


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def create_features(cfg: DictConfig) -> None:
    ResNet152Features(cfg.features.resnet152.config, cfg.data).create_features()


if __name__ == "__main__":
    create_features()  # pylint: disable=no-value-for-parameter
