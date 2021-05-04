import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from vit.embeddings import (
    PatchEmbeddings,
    CLSToken,
    PositionalEmbeddings
)
from vit.transformer import (
    Transformer,
    TransformerSimple
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
        num_layers: int = 8,
        num_heads: int = 16,
        head_dim: int = 128,
        hidden_dim: int = 128,
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
        self.transformer = Transformer(
            dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim
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


class ViTSimple(nn.Module):
    """
    Self contained implementation of ViT
    Note: Uses PyTorch implementation of TransformerEncoder
    """
    def __init__(
        self,
        image_size: int = 256,
        channels: int = 3,
        num_classes: int = 10,
        patch_size: int = 32,
        emb_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 16,
        head_dim: int = 128,
        hidden_dim: int = 512,
        pool: str = "mean"
    ):
        super().__init__()
        self.pool = pool
        patch_dim = channels * patch_size ** 2
        num_patches = (image_size // patch_size) ** 2

        # Embeddings
        self.patch_embeddings = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) c p1 p2", p1=patch_size, p2=patch_size),
            nn.Flatten(start_dim=2),
            nn.Linear(patch_dim, emb_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos = nn.Parameter(torch.randn(num_patches+1, emb_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b = x.shape[0]

        # Embeddings
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos

        # Transformer
        x = self.transformer(x)

        # Classify
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.classifier(x)

        return x