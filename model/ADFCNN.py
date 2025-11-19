import math
import torch
import torch.nn as nn
from model.layers import Conv2dWithConstraint, LazyLinearWithConstraint, PositionalEncodingFourier

class ADFCNN(nn.Module):
    def __init__(self,
                num_channels: int,
                sampling_rate: int,
                F1=8, D=1, F2= 'auto', P1=4, P2=8, pool_mode= 'mean',
                drop_out=0.3, layer_scale_init_value = 1e-6, nums = 4):
        super(ADFCNN, self).__init__()

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        pooling_size = 0.3
        hop_size = 0.7

        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding='same',  max_norm=2.),
            nn.BatchNorm2d(F1),
            )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 30], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
            )

        # Spatial
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (num_channels, 1), padding=0, groups=F2, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            # nn.ELU(),
            nn.SiLU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            # nn.ELU(),
            nn.SiLU(),
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[num_channels, 1], padding='valid',
                                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),

        )

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(drop_out)
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)
        self.flatten =nn.Flatten()

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)  #### the last one is channel

        B, N, C = x_attention.shape

        q = self.w_q(x_attention).permute(0,2,1)
        k = self.w_k(x_attention).permute(0,2,1)
        v = self.w_v(x_attention).permute(0,2,1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        # -------------------
        attn = attn.softmax(dim=-1)

        x = (attn @ v).reshape(B, N, C)

        x_attention = x_attention + self.drop(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
        x = self.drop(x_attention)
        return x

# 原始匹配751个时间点的分类器
class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(8, num_classes, (1, 51)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)  # [16, 4, 1, 19]，维度序号[0,1,2,3]
        x = torch.squeeze(x, 2)  # 先移除height维度 [16, 4, 19]
        # 在时间维度上取平均或最后一个时间点
        x = x.mean(dim=2)  # [16, 4] - 时间维度平均
        # 或者: x = x[:, :, -1]  # [16, 4] - 取最后一个时间点
        return x

#自适应池化的分类器
# class classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(classifier, self).__init__()
#
#         # 使用自适应池化，避免维度问题
#         self.dense = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # 自适应到 [batch, 8, 1, 1]
#             nn.Flatten(),  # 变成 [batch, 8]
#             nn.Linear(8, num_classes),  # [batch, 4]
#             nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.dense(x)

class Net(nn.Module):   
    def __init__(self,
                num_classes: 4,
                num_channels: int,
                sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = ADFCNN(num_channels=num_channels, sampling_rate=sampling_rate)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_model(args):
    
    model = Net(num_classes=args.num_classes,
                num_channels=args.num_channels,
                sampling_rate=args.sampling_rate)

    return model



class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))
