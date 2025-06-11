import torch
import utils.components as components

class Generator(torch.nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        # input: noise_dim x 1 x 1 -> 4x4
        self.conv1 = torch.nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.act1 = torch.nn.ReLU(True)
        self.noise1 = components.NoiseInjection(1024)

        # 4x4 -> 8x8
        self.conv2 = torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.act2 = torch.nn.ReLU(True)
        self.noise2 = components.NoiseInjection(512)

        # 8x8 -> 16x16
        self.conv3 = torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.act3 = torch.nn.ReLU(True)
        self.noise3 = components.NoiseInjection(256)

        # 16x16 -> 32x32
        self.conv4 = torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.act4 = torch.nn.ReLU(True)
        self.noise4 = components.NoiseInjection(128)

        # Refinement at 32x32
        self.refine_conv = torch.nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.refine_bn = torch.nn.BatchNorm2d(64)
        self.refine_act = torch.nn.ReLU(True)
        self.noise_refine = components.NoiseInjection(64)

        # 32x32 -> 64x64
        self.conv5 = torch.nn.ConvTranspose2d(64, 16, 4, 2, 1, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(16)
        self.act5 = torch.nn.ReLU(True)
        self.noise5 = components.NoiseInjection(16)

        # Output 64x64x3
        self.conv6 = torch.nn.Conv2d(16, 3, 3, 1, 1, bias=False)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.noise1(self.act1(self.bn1(self.conv1(x))))
        x = self.noise2(self.act2(self.bn2(self.conv2(x))))
        x = self.noise3(self.act3(self.bn3(self.conv3(x))))
        x = self.noise4(self.act4(self.bn4(self.conv4(x))))
        x = self.noise_refine(self.refine_act(self.refine_bn(self.refine_conv(x))))
        x = self.noise5(self.act5(self.bn5(self.conv5(x))))
        x = self.tanh(self.conv6(x))
        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 64x64x3 -> 32x32x64
        self.init_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )
        # 32x32x64
        
        self.down0 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )
        

        self.down1 = components.Module(
            conv_blocks_number=0,
            in_channels=128,
            internal_channels=128,
            out_channels=128,
            bypass=True,
            max_pool=True,
            batch_norm=True,
            dropout=False
        )
        self.down2 = components.Module(
            conv_blocks_number=0,
            in_channels=128,
            internal_channels=256,
            out_channels=256,
            bypass=True,
            max_pool=True,
            batch_norm=True,
            dropout=False
        )
        self.down3 = components.Module(
            conv_blocks_number=1,
            in_channels=256,
            internal_channels=512,
            out_channels=512,
            bypass=True,
            max_pool=True,
            batch_norm=True,
            dropout=False
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = torch.nn.Linear(512, 1)

    def features(self, x):
        x = self.init_block(x)
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.init_block(x)
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x