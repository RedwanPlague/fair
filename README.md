# FAIR

## ECCV'26 Paper: [FAIR: Feature-Augmented Implicit Regularization for AI-generated Fake Image Detection](https://arxiv.org/abs/2607.22087)

SCS Extraction Code: https://github.com/RedwanPlague/scssim

You really only need the SCS feature extraction for FAIR. There's not really any complicated code. Just `torch.cat()` the existing features with SCS features and increase the dimension of the classifier's first `nn.Linear` layer. I've added a Pseudo code. Will upload my additions to AIDE and PatchCraft soon.
