import torch
import torch.nn as nn


# Replace with any detector
class BaseDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


if __name__ == "__main__":
    print(BaseDetector()(torch.randn(1, 1024)))
