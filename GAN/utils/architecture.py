import torch
import utils.components as components

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.blocks = torch.nn.Sequential(
                *[
                    components.TransposedConvolutionalBlock(
                            in_channels = 100, 
                            out_channels = 512,
                            kernel_size = 4, 
                            stride = 1, 
                            padding = 0,
                            batch_norm = True,
                            activation = 'relu'
                    ),
                    components.TransposedConvolutionalBlock(
                            in_channels = 512, 
                            out_channels = 256,
                            kernel_size = 4, 
                            stride = 2, 
                            padding = 1,
                            batch_norm = True,
                            activation = 'relu'
                    ),
                    components.TransposedConvolutionalBlock(
                            in_channels = 256, 
                            out_channels = 128,
                            kernel_size = 4, 
                            stride = 2, 
                            padding = 1,
                            batch_norm = True,
                            activation = 'relu'
                    ),
                    components.TransposedConvolutionalBlock(
                            in_channels = 128, 
                            out_channels = 64,
                            kernel_size = 4, 
                            stride = 2, 
                            padding = 1,
                            batch_norm = True,
                            activation = 'relu'
                    ),
                    components.TransposedConvolutionalBlock(
                            in_channels = 64, 
                            out_channels = 3,
                            kernel_size = 4, 
                            stride = 2, 
                            padding = 1,
                            batch_norm = True,
                            activation = 'tanh'
                    )
                ]
            )

    def forward(self, x):
        x = torch.nn.Identity(x)
        for it in self.blocks:
            x = it(x)
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.init_block = components.InitBlock(out_channels = 64)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = True,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 128, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = False,
                        batch_norm = True,
                        dropout = False
                    )
        ]) 
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.init_block(x)
        # this relu does nothing ?
        x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x