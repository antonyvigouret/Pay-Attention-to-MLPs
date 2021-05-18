import einops
import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_channel, d_spatial):
        super().__init__()

        self.d_channel = d_channel
        self.d_spatial = d_spatial

        self.norm = nn.LayerNorm(d_channel // 2)
        self.proj = nn.Linear(d_spatial, d_spatial)

    def forward(self, x):
        u, v = torch.split(x, self.d_channel // 2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, d_ffn, d_spatial):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, d_spatial)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)

    def forward(self, x):
        shorcut = x
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut


class GatingMlp(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        d_spatial,
        n_blocks,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [GatingMlpBlock(d_model, d_ffn, d_spatial) for _ in range(n_blocks)]
        )

    def forward(self, x):
        for gmlp_block in self.blocks:
            x = gmlp_block(x)
        return x


class VisionGatingMlp(nn.Module):
    def __init__(
        self,
        image_size,
        n_channels,
        patch_size,
        d_model,
        d_ffn,
        n_blocks,
        n_classes,
    ):
        super().__init__()

        assert image_size % patch_size == 0
        self.n_patches = (image_size // patch_size) ** 2
        self.d_spatial = self.n_patches + 1

        self.patch_embedding = nn.Conv2d(
            n_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gmlp = GatingMlp(d_model, d_ffn, self.d_spatial, n_blocks)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]

        x = self.patch_embedding(x)
        x = einops.rearrange(x, "n c h w -> n (h w) c")
        
        cls_token = self.cls_token.expand(n_samples, 1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.gmlp(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

if __name__ == "__main__":
    image_size = 224
    n_channels = 3
    patch_size = 16
    n_classes = 10
    
    d_model = 128
    d_ffn = 768
    n_blocks = 30
    
    args = {
        "image_size": image_size,
        "n_channels": n_channels,
        "patch_size": patch_size,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "n_blocks": n_blocks,
        "n_classes": n_classes,
    }

    vi_gmlp = VisionGatingMlp(**args)
    total_params = sum(p.numel() for p in vi_gmlp.parameters() if p.requires_grad)
    print(total_params)
    
    x = torch.rand((2, 3, 224, 224))
    y = vi_gmlp(x)
    print(y.shape)