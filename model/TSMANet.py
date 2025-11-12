import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Multi-scale 1D Convolution Module
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p) for k, p in zip(kernel_sizes, padding)
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        out = self.dropout(out)
        return out


# Multi-Headed Attention Module with Local and Global Attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head

        kernel_sizes = [3, 5]
        padding = [1, 2]

        self.multi_scale_conv_k = MultiScaleConv1d(d_model, d_model, kernel_sizes, padding)

        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k_local = nn.Linear(d_model * len(kernel_sizes), n_head * self.d_k)
        self.w_k_global = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        bsz = query.size(0)

        key_local = key.transpose(1, 2)
        key_local = self.multi_scale_conv_k(key_local).transpose(1, 2)

        q = self.w_q(query).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        k_local = self.w_k_local(key_local).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        k_global = self.w_k_global(key).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bsz, -1, self.n_head, self.d_v).transpose(1, 2)

        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = self.dropout(attn_local)
        x_local = torch.matmul(attn_local, v)

        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = self.dropout(attn_global)
        x_global = torch.matmul(attn_global, v)

        x = x_local + x_global

        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.n_head * self.d_v)
        return self.w_o(x)


# Feed-Forward Neural Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output


# Feature Extraction Module
class ExtractFeature(nn.Module):
    def __init__(self, num_channels, num_samples, embed_dim, pool_size, pool_stride):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(embed_dim)

        self.spatial_conv1 = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1), padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.glu = nn.GELU()
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x = x1 + x2
        x = self.bn1(x)
        x = self.spatial_conv1(x)
        x = self.glu(x)
        x = self.bn2(x)
        x = x.squeeze(dim=2)
        x = self.avg_pool(x)
        return x


# Transformer Module
class TransformerModule(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop):
        super().__init__()
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)
        ])

    def forward(self, x):
        x = rearrange(x, 'b d n -> b n d')
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=2)
        return x


# Classification Module
class ClassifyModule(nn.Module):
    def __init__(self, embed_dim, temp_embedding_dim, num_classes):
        super().__init__()
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.classify(x)
        return out


# Complete TMSA-Net Model
class TMSANet(nn.Module):
    def __init__(self, num_channels, sampling_rate, num_classes, embed_dim=19, pool_size=8,
                 pool_stride=8, num_heads=4, fc_ratio=4, depth=3, attn_drop=0.5, fc_drop=0.5):
        super().__init__()

        # 计算时间点数（根据采样率，假设1秒数据）
        time_points = sampling_rate  # 或者根据实际需求调整

        self.in_planes = num_channels
        self.extract_feature = ExtractFeature(self.in_planes, time_points, embed_dim, pool_size, pool_stride)
        temp_embedding_dim = (time_points - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout()
        self.transformer_module = TransformerModule(embed_dim, num_heads, fc_ratio, depth, attn_drop, fc_drop)
        self.classify_module = ClassifyModule(embed_dim, temp_embedding_dim, num_classes)

    def forward(self, x):
        # 输入x的形状应该是 [batch, 1, channels, time_points]
        # 但为了兼容性，我们需要处理不同的输入形状
        if x.dim() == 4:
            # 如果输入是4D [batch, 1, channels, time_points]
            x = x.squeeze(1)  # 移除通道维度 -> [batch, channels, time_points]

        x = self.extract_feature(x)
        x = self.dropout(x)
        x = self.transformer_module(x)
        out = self.classify_module(x)
        return out


# 提供统一模型调用接口（与ADFCNN保持一致）
def get_model(args):
    model = TMSANet(
        num_channels=args.num_channels,
        sampling_rate=args.sampling_rate,
        num_classes=args.num_classes
    )
    return model


# 可选：添加Net包装类以保持完全一致的结构
class Net(nn.Module):
    def __init__(self, num_classes, num_channels, sampling_rate):
        super(Net, self).__init__()
        self.backbone = TMSANet(
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


# 提供两种接口方式
def get_model(args):
    # 方式1：直接使用TMSANet
    model = TMSANet(
        num_channels=args.num_channels,
        sampling_rate=args.sampling_rate,
        num_classes=args.num_classes
    )

    # 方式2：使用Net包装（与ADFCNN完全一致）
    # model = Net(
    #     num_classes=args.num_classes,
    #     num_channels=args.num_channels,
    #     sampling_rate=args.sampling_rate
    # )

    return model