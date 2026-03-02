import torch
from torch import nn

from .attention import Attention, CrossAttention, get_timestep_embedding, PVCNN2SA, PVCNN2FP
from .graph_wavelet_transform import batch_chebyshev_wavelet_transform


class MoE(nn.Module):
    def __init__(self, dim, mlp_dim, drop_rate, centers):
        super(MoE, self).__init__()

        self.expert_1 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop_rate)
        )
        self.expert_2 = GeometricExpert(dim, centers)
        self.w = nn.Parameter(torch.zeros(centers, 1))

    def forward(self, x):
        """
        x: [B, N, C]  point features
        """
        # Base expert output
        y1 = self.expert_1(x)

        y2 = self.expert_2(x)

        w = torch.sigmoid(self.w).unsqueeze(0)  # [1, N, 1]
        return y1 * w + (1 - w) * y2

class GeometricExpert(nn.Module):
    def __init__(self, dim, centers, ):
        super().__init__()
        self.map = nn.Parameter(torch.randn(centers, dim))
        self.ada = nn.Linear(dim, 2 * dim)
        nn.init.normal_(self.map, mean=0, std=0.01)

    def forward(self, x):
        g = x * self.map
        g = self.ada(g)
        alpha, beta = torch.chunk(g, 2, dim=-1)
        return x * alpha + beta


class Block(nn.Module):

    def __init__(self, dim=768, mlp_dim=3072, num_heads=8, drop_rate=0.0, wave_attn_dim=32, centers=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, proj_drop=drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_image = CrossAttention(dim=dim, proj_drop=0.1)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MoE(dim, mlp_dim, drop_rate, centers)

        self.wave_down = nn.Linear(dim, wave_attn_dim)
        self.wave_norm1 = nn.LayerNorm(wave_attn_dim)
        self.wave_attn_image = CrossAttention(dim=wave_attn_dim, kv_in_dim=dim, proj_drop=0.1)
        self.wave_norm2 = nn.LayerNorm(wave_attn_dim)
        self.wave_up = nn.Sequential(
            nn.Linear(wave_attn_dim, dim),
        )
        self.wave_ada = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x, image_feats, wave_basis):
        x = x + self.attn(self.norm1(x))
        x = x + self.attn_image(self.norm2(x), image_feats)

        wave_x = wave_basis @ x
        wave_x_snapshot = wave_x
        wave_x = self.wave_down(wave_x)
        wave_x = wave_x + self.wave_attn_image(self.wave_norm1(wave_x), image_feats)
        wave_x = self.wave_ada(wave_x_snapshot + self.wave_up(self.wave_norm2(wave_x)))
        x = x + wave_x
        x = x + self.mlp(self.norm3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, dim=768, mlp_dim=3072, num_heads=8, drop_rate=0.0, depth=4, wave_attn_dim=64, centers=512):
        super().__init__()
        self.depth = depth
        blocks = [Block(dim=dim, mlp_dim=mlp_dim, num_heads=num_heads, drop_rate=drop_rate, wave_attn_dim=wave_attn_dim,
                        centers=centers) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, image_feats, wave_basis):
        for block in self.blocks:
            x = block(x, image_feats, wave_basis)
        return x


class WGDMNet(nn.Module):
    def __init__(self, dim=768, depth=12, mlp_dim=3072, num_heads=8, drop_rate=0.1, num_centers=512, num_neighbour=32,
                 extra_feature_channels=0, use_surface_project=False, projection_dim=64, wave_attn_dim=64):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.drop_rate = drop_rate
        self.use_surface_project = use_surface_project
        if self.use_surface_project:
            self.projection_in_proj = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.GELU(),
                nn.Linear(128, projection_dim)
            )
            extra_feature_channels += projection_dim
        self.pointnet_sa = PVCNN2SA(
            extra_feature_channels=extra_feature_channels,
            dropout=0.1, width_multiplier=1,
            voxel_resolution_multiplier=1,
            dim=dim,
            num_centers=num_centers,
            num_neighbour=num_neighbour
        )
        self.pointnet_fp = PVCNN2FP(
            num_classes=3,
            sa_in_channels=self.pointnet_sa.sa_in_channels,
            channels_sa_features=self.pointnet_sa.channels_sa_features,
            extra_feature_channels=extra_feature_channels,
            dropout=0.1, width_multiplier=1,
            voxel_resolution_multiplier=1
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.dim)
        )
        self.encoder = Encoder(dim=self.dim, depth=int(self.depth), mlp_dim=self.mlp_dim, num_heads=num_heads,
                               drop_rate=self.drop_rate, wave_attn_dim=wave_attn_dim, centers=num_centers)

    def forward(self, point_cloud, mvp_token, image_feats, projected_feats, time_step):
        point_cloud_feats = point_cloud
        if self.use_surface_project:
            projected_feats = self.projection_in_proj(projected_feats)
            point_cloud_feats = torch.cat([point_cloud_feats, projected_feats], dim=-1)
        in_features_list, coords_list, group_input_tokens, center, knn = self.pointnet_sa(point_cloud_feats)
        wave_basis = batch_chebyshev_wavelet_transform(center.transpose(-1, -2).contiguous(), k=12)
        group_input_tokens = group_input_tokens.transpose(-1, -2).contiguous()
        # time_step_encode
        time_embedding = get_timestep_embedding(self.dim, time_step, time_step.device)
        time_condition = time_embedding.unsqueeze(1)
        # add pos embedding
        pos_point = self.pos_embed(center.transpose(-1, -2).contiguous())
        # condition
        # mvp_token = torch.cat((time_condition, mvp_token), dim=1)
        image_feats = torch.cat((time_condition, image_feats), dim=1)
        # transformer
        x = self.encoder(group_input_tokens + pos_point, image_feats, wave_basis)
        return self.pointnet_fp(in_features_list, coords_list, x.transpose(-1, -2).contiguous(), center)
