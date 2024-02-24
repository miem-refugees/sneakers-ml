from collections.abc import Sequence

import numpy as np
import onnxruntime
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
    def __init__(self, config: DictConfig, config_data: DictConfig) -> None:
        super().__init__(config, config_data)

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.device = self.get_device(self.config.device)

        if self.config.use_onnx is True:
            self.providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            )
            self.onnx_session = onnxruntime.InferenceSession(self.config.onnx_path, providers=self.providers)
        else:
            self.model = self.initialize_torch_resnet()
            self.model.to(self.device)

    def initialize_torch_resnet(self) -> torch.nn.Module:
        model = resnet152(weights=self.weights)
        model.fc = Identity()
        model.eval()
        return model  # type: ignore[no-any-return]

    def create_onnx_model(self) -> None:
        model = self.initialize_torch_resnet()
        torch_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            torch_input,
            self.config.onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def get_feature(self, image: Image.Image) -> np.ndarray:
        return np.squeeze(self.get_features([image]))

    def get_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])

        if self.config.use_onnx is True:
            onnxruntime_input = {
                self.onnx_session.get_inputs()[0].name: np.array([self.to_numpy(x) for x in preprocessed_images])
            }
            return self.onnx_session.run(["output"], onnxruntime_input)[0]  # type: ignore[no-any-return]

        with torch.inference_mode():
            x = preprocessed_images.to(self.device)
            prediction = self.model(x)
        return prediction.cpu().numpy()  # type: ignore[no-any-return]

    def get_features_folder(self, folder_path: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        if self.config.use_onnx is True:
            raise NotImplementedError

        dataset = ImageFolder(folder_path, transform=self.preprocess)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

        features = []
        with torch.inference_mode():
            for data in tqdm(dataloader):
                x = data[0].to(self.device)
                prediction = self.model(x)

                features.append(prediction.cpu())

        full_images_features = torch.cat(features, dim=0)
        numpy_features = full_images_features.numpy()
        classes = np.array(dataset.imgs)

        return numpy_features, classes, dataset.class_to_idx
