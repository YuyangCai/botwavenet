import torch
from torch import nn
from torchvision.models import resnet50
from bottleneck_transformer_pytorch import BottleStack

class BotNetEncoder(nn.Module):
    def __init__(self, in_channels=3, bot_layer_params=None, skip2_out_channels=512):
        super().__init__()
        # Load pre-trained ResNet50 model
        self.resnet = resnet50(pretrained=True)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.skip2_conv = nn.Conv2d(bot_layer_params['dim_out'], skip2_out_channels, kernel_size=1)
        # BotNet layer parameters
        if bot_layer_params is None:
            bot_layer_params = {
                'dim': 2048,
                'fmap_size': 8,  # size at which to apply the Bottleneck Transformer
                'dim_out': 2048,
                'proj_factor': 4,
                'downsample': False,
                'heads': 4,
                'dim_head': 128,
                'rel_pos_emb': True,
                'activation': nn.ReLU()
            }

        self.bot_layer = BottleStack(**bot_layer_params)

        # Model surgery
        self.initial_layers = nn.Sequential(*list(self.resnet.children())[:7])  # Up to layer3
        self.layer4 = list(self.resnet.layer4)[:-1]  # All of layer4 except its last Bottleneck
        self.final_layers = nn.Sequential(
            *self.layer4,
            list(self.resnet.layer4)[-1],  # Last Bottleneck of layer4
            self.bot_layer,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_channels = bot_layer_params['dim_out']  # This should be 2048 in your case

    def forward(self, x):
        # Initial layers
        x = self.initial_layers(x)
        skip1 = x

        # Final layers
        x = self.final_layers(x)
        x = self.skip2_conv(x)
        skip2 = x  # You might need to adjust which layer's output to use as skip connection

        return [skip1, skip2], x

# Instantiate the encoder
encoder = BotNetEncoder(skip2_out_channels=512)

