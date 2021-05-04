import einops
import torch
import torch.nn as nn



class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)

        # No projection to original dim necessary
        if num_heads == 1 and dim == head_dim:
            self.proj = nn.Identity()
        # Project to original dim
        else:
            self.proj = nn.Linear(self.inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [bs, num_seq, dim]
        b, n, d = x.shape

        # Project [bs, num_seq, num_heads * head_dim * 3]
        qkv = self.qkv(x)

        # Break into q, k, v tuple [bs, num_seq, num_heads * head_dim]
        qkv = qkv.chunk(3, dim=-1)

        # Reshape to [bs, num_heads, num_seq, head_dim]
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        # Dot product of queries and keys
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # Scale scores [bs, num_heads, num_seq, num_seq] (similarity matrix)
        scores = scores * self.scale

        # Normalize scores to pdist [bs, num_heads, num_seq, num_seq]
        attn = self.attn(scores)

        # Apply attention to values [bs, num_heads, num_seq, num_seq]
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        # Reshape to [bs, num_seq, num_heads * head_dim]
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        # Project to input dim if necessary
        out = self.proj(out)

        return out


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Transformer(nn.Module):

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Attention(dim, head_dim, num_heads)),
                nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, hidden_dim))
            ]
            self.layers.append(nn.ModuleList(layer))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerSimple(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
