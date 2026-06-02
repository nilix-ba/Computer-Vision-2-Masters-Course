import torch
import torch.nn as nn
import torchvision.models as models

class BiologicallyOptimizedSaliencyNet(nn.Module):
    def __init__(self):
        super(BiologicallyOptimizedSaliencyNet, self).__init__()
        
        # Ultra-lightweight backbone encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Split ResNet into progressive stages to harvest multi-level features
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # Out: 64 ch, 56x56
        self.layer1 = resnet.layer1  # Out: 64 ch, 56x56 (Low-level boundaries)
        self.layer2 = resnet.layer2  # Out: 128 ch, 28x28
        self.layer3 = resnet.layer3  # Out: 256 ch, 14x14
        self.layer4 = resnet.layer4  # Out: 512 ch, 7x7 (High-level semantics)
        
        # 2. Resource-minimal Decoder using Skip Connections
        # Instead of throwing away spatial data, we reuse early encoder layers
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 7x7 -> 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), # 512 comes from (256 upsampled + 256 skip from layer3)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 14x14 -> 28x28
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # (128 upsampled + 128 skip from layer2)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 28x28 -> 56x56
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # (64 upsampled + 64 skip from layer1)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling stages to get back to full 224x224 resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 112x112 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)                       # Final single-channel map
        )
        
        # 3. Biologically Inspired Learnable Center Bias Prior
        # A single 1x224x224 parameter initialized as a Gaussian distribution.
        # This adds zero heavy convolutions but massively boosts fixation accuracy.
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 224, 224))
        self._initialize_center_bias()

    def _initialize_center_bias(self):
        # Create a static 2D Gaussian matrix centered in the middle of the image frame
        y, x = torch.meshgrid(torch.linspace(-1, 1, 224), torch.linspace(-1, 1, 224), indexing='ij')
        gaussian = torch.exp(-0.5 * (x**2 + y**2) / (0.5**2)) # Variance scale of 0.5
        self.center_bias.data.copy_(gaussian.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # --- Encoder Forward Pass ---
        x0 = self.init_conv(x)      # 56x56
        x1 = self.layer1(x0)        # 56x56  <- Low-level Skip target
        x2 = self.layer2(x1)        # 28x28  <- Mid-level Skip target
        x3 = self.layer3(x2)        # 14x14  <- High-level Skip target
        x4 = self.layer4(x3)        # 7x7 Deep bottleneck
        
        # --- Decoder Forward Pass with Concat Skip Connections ---
        u1 = self.up1(x4)
        merged1 = torch.cat([u1, x3], dim=1) # Combine deep semantics with sharp spatial structures
        c1 = self.conv1(merged1)
        
        u2 = self.up2(c1)
        merged2 = torch.cat([u2, x2], dim=1)
        c2 = self.conv2(merged2)
        
        u3 = self.up3(c2)
        merged3 = torch.cat([u3, x1], dim=1)
        c3 = self.conv3(merged3)
        
        # Generate raw visual prediction map
        saliency_map = self.final_upsample(c3)
        
        # --- Inject Center Bias ---
        # Simply adding the learnable center tracking grid directly to the raw output logits
        output = saliency_map + self.center_bias
        
        return torch.sigmoid(output)