import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
import torch.optim as optim


to_tensor_transform = [Resize((144, 144)), ToTensor()]


class Transform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target):
        for t in self.transform:
            img = t(img)
        return img, target


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Attention(nn.Module):
    def __init__(self, dim=128, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim
        self.multi_head_attention = nn.MultiheadAttention(
            dim, num_heads)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q(x)
        k = self.q(x)
        v = self.q(x)

        return self.multi_head_attention(q, k, v)[0]


class LayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.layers(x)


class ViT(pl.LightningModule):
    def __init__(self, img_size=144, num_layers=6, num_heads=4, patch_size=4, channels=3, embed_dim=768, num_classes=37):
        super().__init__()
        self.save_hyperparameters()

        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(patch_size, channels, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer = nn.ModuleList([
            nn.Sequential(
                Residual(LayerNorm(embed_dim, Attention(embed_dim, num_heads))),
                Residual(LayerNorm(embed_dim, MLP(in_features=embed_dim, hidden_features=embed_dim * 4, out_features=embed_dim)))
            ) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.patch_embedding(x)
        x = x + self.positional_embedding
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        for layer in self.transformer:
            x = layer(x)
        return self.classifier(x[:, 0, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer
