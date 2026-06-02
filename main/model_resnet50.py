import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50SaliencyNet(nn.Module):
    def __init__(self):
        super(ResNet50SaliencyNet, self).__init__()
        
        # Load the deeper ResNet-50 backbone features
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Isolate the progressive layers to handle expanded channels
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # Out: 64 ch, 56x56
        self.layer1 = resnet.layer1  # Out: 256 ch, 56x56 (ResNet-18 was 64)
        self.layer2 = resnet.layer2  # Out: 512 ch, 28x28 (ResNet-18 was 128)
        self.layer3 = resnet.layer3  # Out: 1024 ch, 14x14 (ResNet-18 was 256)
        self.layer4 = resnet.layer4  # Out: 2048 ch, 7x7   (ResNet-18 was 512)
        
        # --- Adjusted Decoder channels to support ResNet-50 dimensions ---
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2) # 7x7 -> 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(512 + 1024, 256, kernel_size=3, padding=1), # 512 upsampled + 1024 layer3 skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2) # 14x14 -> 28x28
        self.conv2 = nn.Sequential(
            nn.Conv2d(256 + 512, 128, kernel_size=3, padding=1),  # 256 upsampled + 512 layer2 skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # 28x28 -> 56x56
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + 256, 64, kernel_size=3, padding=1),   # 128 upsampled + 256 layer1 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 112x112 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # Biological Center Bias Prior
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 224, 224))
        self._initialize_center_bias()

    def _initialize_center_bias(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 224), torch.linspace(-1, 1, 224), indexing='ij')
        gaussian = torch.exp(-0.5 * (x**2 + y**2) / (0.5**2))
        self.center_bias.data.copy_(gaussian.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x0 = self.init_conv(x)      # 56x56
        x1 = self.layer1(x0)        # 56x56  <- Skip Target 1
        x2 = self.layer2(x1)        # 28x28  <- Skip Target 2
        x3 = self.layer3(x2)        # 14x14  <- Skip Target 3
        x4 = self.layer4(x3)        # 7x7 Deep Bottleneck
        
        u1 = self.up1(x4)
        merged1 = torch.cat([u1, x3], dim=1)
        c1 = self.conv1(merged1)
        
        u2 = self.up2(c1)
        merged2 = torch.cat([u2, x2], dim=1)
        c2 = self.conv2(merged2)
        
        u3 = self.up3(c2)
        merged3 = torch.cat([u3, x1], dim=1)
        c3 = self.conv3(merged3)
        
        saliency_map = self.final_upsample(c3)
        output = saliency_map + self.center_bias
        return torch.sigmoid(output)