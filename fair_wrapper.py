import torch
import torch.nn as nn

from base_detector import BaseDetector


class FAIRWrapper(nn.Module):
    def __init__(self, N_prior, M_prior):
        super().__init__()

        # has a feature_extractor and a classifier
        self.base = BaseDetector()

        self.prior_projector = nn.Sequential(
            nn.Linear(N_prior, M_prior),
            nn.GELU()
        )

        old_layer1 = self.base.classifier[0]
        self.D = old_layer1.in_features

        self.base.classifier[0] = nn.Linear(
            self.D + M_prior, old_layer1.out_features
        )

    def forward(self, image, scs):
        X = self.base.feature_extractor(image)

        X_star = self.prior_projector(scs)
        X_aug = torch.cat([X, X_star], dim=1)

        return self.base.classifier(X_aug)

    def save_model(self, save_path):
        state_dict = self.base.state_dict()

        weight_key = "classifier.0.weight"
        w = state_dict[weight_key]
        state_dict[weight_key] = w[:, :self.D]

        torch.save(state_dict, save_path)


if __name__ == "__main__":
    trainer = FAIRWrapper(1024, 256)
    print(trainer(torch.randn(1, 1024), torch.randn(1, 1024)))
    trainer.save_model("ckpt.pth")
    model = BaseDetector()
    model.load_state_dict(state_dict)
    model(torch.randn(1, 1024))
