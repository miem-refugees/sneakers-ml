import lightning as L
import torch
from torch import nn
from torchvision.models import ResNet152_Weights, resnet152


class ResNet152Classifier(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        weights = ResNet152_Weights.DEFAULT
        backbone = resnet152(weights=weights)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
