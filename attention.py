import torch
import numpy as np

from .pvcnn.pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules
from torch import nn



def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).float()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H N D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # x = einops.rearrange(x, 'B H N D -> B N (H D)')
        x = x.transpose(1, 2).flatten(-2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, q_in_dim=None, kv_in_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        q_in_dim = q_in_dim if q_in_dim is not None else dim
        kv_in_dim = kv_in_dim if kv_in_dim is not None else dim
        self.q = nn.Linear(q_in_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_in_dim, 2 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).float()
        kv = self.kv(c).reshape(B, c.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).float()
        k, v = kv[0], kv[1]  # B H N D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).flatten(-2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PVCNN2SA(nn.Module):

    def __init__(self, use_att=True, dropout=0.1, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, dim=384, num_centers=512, num_neighbour=32):
        super().__init__()
        assert extra_feature_channels >= 0
        self.sa_blocks = [
            ((32, 2, 32), (num_centers, 0.1, num_neighbour, (128, dim)))
        ]
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        self.in_channels = extra_feature_channels + 3

        # Create PointNet-2 model
        sa_layers, self.sa_in_channels, self.channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, inputs):
        inputs = inputs.transpose(-1, -2).contiguous()
        # Separate input coordinates and features
        coords = inputs[:, :3, :].contiguous()  # (B, 3, N)
        features = inputs  # (B, 3 + S, N)

        # Downscaling layers
        coords_list = []
        in_features_list = []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords, knn = sa_blocks((features, coords))
        # Replace the input features
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        return in_features_list, coords_list, features, coords, knn


class PVCNN2FP(nn.Module):
    fp_blocks = [
        ((128,), (32, 2, 32))
    ]

    def __init__(self, num_classes, sa_in_channels, channels_sa_features, use_att=True, dropout=0.1,
                 extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, in_features_list, coords_list, features, coords):
        # Upscaling layers
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks(
                (  # this is a tuple because of nn.Sequential
                    coords_list[-1 - fp_idx],  # reverse coords list from above
                    coords,  # original point coordinates
                    features,  # keep concatenating upsampled features
                    in_features_list[-1 - fp_idx],  # reverse features list from above
                )
            )
        # Output MLP layers
        output = self.classifier(features)
        return output.transpose(1, 2).contiguous()


def modulate(x, shift, scale):
    return x * scale + shift
