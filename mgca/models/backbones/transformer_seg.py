import logging
import math
import os
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
from mgca.models.backbones.transformer_model import TransModel2d, TransConfig
import math


class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * \
            config.hidden_size % (sample_v**2) == 0, "不能除尽"
        self.final_dense = nn.Linear(
            config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // (sample_v**2))
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v

        self.is_segmentation = is_segmentation

    def forward(self, x):
        # x:(b, c, w, h)
        b, c, h, w = x.shape
        # assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

        # if h % p1 != 0:
        #     print("请重新输入img size 参数 必须整除")
        #     os._exit(0)
        # if w % p2 != 0:
        #     print("请重新输入img size 参数 必须整除")
        #     os._exit(0)
        hh = h // p1
        ww = w // p2

        # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2)

        # encode_x = self.bert_model(x)[-1] # 取出来最后一层
        encode_x = self.bert_model(x)[:, 1:]
        if not self.is_segmentation:
            return encode_x

        x = self.final_dense(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      p1=self.hh, p2=self.ww, h=hh, w=ww, c=self.config.hidden_size)
        return encode_x, x


class PreTrainModel(nn.Module):
    def __init__(self, patch_size,
                 in_channels,
                 out_class,
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=0,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out


class Vit(nn.Module):
    def __init__(self, patch_size,
                 in_channels,
                 out_class,
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 sample_rate=4,
                 ):
        super().__init__()
        config = TransConfig(patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=0,
                             sample_rate=sample_rate,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)

        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out


class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x


class SETRModel(nn.Module):
    def __init__(self, patch_size=(32, 32),
                 in_channels=3,
                 out_channels=1,
                 hidden_size=1024,
                 num_hidden_layers=8,
                 num_attention_heads=16,
                 decode_features=[512, 256, 128, 64],
                 sample_rate=4,):
        super().__init__()
        config = TransConfig(patch_size=patch_size,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             sample_rate=sample_rate,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size,
                                    out_channels=config.out_channels, features=decode_features)

    def forward(self, x):
        _, final_x = self.encoder_2d(x)
        x = self.decoder_2d(final_x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 3, 224, 224)
    net = SETRModel(
        patch_size=(16, 16),
        in_channels=3,
        out_channels=1,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        decode_features=[512, 256, 128, 64]
    )
    out = net(x)