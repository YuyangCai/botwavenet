import torch.nn as nn
from torchvision.models import resnet50
from bottleneck_transformer_pytorch import BottleStack

class BotNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Initialize the resnet
        self.resnet = resnet50(pretrained=True)
        
        # Modify the ResNet to include the BottleStack
        self.resnet.layer4 = BottleStack(
            dim=1024,
            fmap_size=32,  # Adjust size to match feature map.
            dim_out=2048,
            proj_factor=4,
            downsample=True,
            heads=4,
            dim_head=128,
            rel_pos_emb=True,
            activation=nn.ReLU()
        )
        self.out_channels = [256, 512, 1024, 2048]  # Example, adjust based on your layers

    def forward(self, x):
        # Implement forward pass, returning a list of feature maps
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        features.append(x)

        x = self.resnet.layer3(x)
        features.append(x)

        x = self.resnet.layer4(x)
        features.append(x)
        print(x.shape)
        
        return features