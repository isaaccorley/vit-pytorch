import torch
import torch.nn as nn

from vit.embeddings import (
    PatchEmbeddings,
    CLSToken,
    PositionalEmbeddings
)
from vit.transformer import (
    TransformerEncoder,
    TransformerEncoderSimple
)



class Pooling(nn.Module):

    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(
        self,
        dim: int,
        num_classes: int
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ViT(nn.Module):
    
    def __init__(
        self,
        image_size: int = 256,
        channels: int = 3,
        num_classes: int = 10,
        patch_size: int = 32,
        emb_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 8,
        pool: str = "mean"
    ):
        super().__init__()
        patch_dim = channels * patch_size ** 2
        num_patches = (image_size // patch_size) ** 2
        self.patch_embeddings = PatchEmbeddings(
            patch_size=patch_size,
            patch_dim=patch_dim,
            emb_dim=emb_dim
        )
        self.cls_token = CLSToken(dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(
            num_pos=num_patches + 1, # +1 for cls token
            dim=emb_dim
        )
        self.transformer = TransformerEncoderSimple(
            dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.patch_embeddings(x)
        x = self.cls_token(x)
        x = self.pos_embeddings(x)
        x = self.transformer(x)
        x = self.pool(x)
        return self.classifier(x)
