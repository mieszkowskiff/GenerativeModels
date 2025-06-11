import torch
import tqdm
from sklearn.metrics import f1_score

class InitBlock(torch.nn.Module):
    def __init__(self, out_channels = 32):
        super(InitBlock, self).__init__()
        '''
        self.init_conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, out_channels, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.BatchNorm2d(out_channels, affine=True) 
            )  
        '''      
        self.init_conv = torch.nn.Conv2d(3, out_channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        #y = self.init_conv(x)
        #return torch.nn.functional.relu(x + y, inplace = True)
        x = self.init_conv(x)
        return torch.nn.functional.relu(x, inplace = True)

class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, 
                 out_channels = 32, 
                 bypass = False, 
                 batch_norm = False,
                 activation = 'relu'
                 ): 
        super(ConvolutionalBlock, self).__init__()
        self.bypass = bypass
        self.batch_norm = batch_norm
        self.activation = activation
        if(self.batch_norm):
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.BatchNorm2d(out_channels, affine=True) 
            )        
        else:
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            )

    def forward(self, x):
        y = self.block(x)
        if self.activation == 'tanh':
            if self.bypass:
                return torch.nn.functional.tanh(x + y)
            return torch.nn.functional.tanh(y)
        if self.bypass:
            return torch.nn.functional.relu(x + y, inplace = True)
        return torch.nn.functional.relu(y, inplace = True)
        

class TransposedConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels=32, 
                 out_channels=32, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1,                  
                 batch_norm=False, 
                 activation='relu',
                 use_residual=True):  # NEW
        super(TransposedConvolutionalBlock, self).__init__()
        self.use_residual = use_residual
        self.batch_norm = batch_norm
        self.activation = activation

        layers = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
        if self.batch_norm:
            layers.append(torch.nn.BatchNorm2d(out_channels, affine=True))
        self.block = torch.nn.Sequential(*layers)

        # If shapes differ, prepare a 1x1 conv to align residual connection
        self.shortcut = None
        if use_residual and (in_channels != out_channels or stride != 1):
            self.shortcut = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        residual = x
        y = self.block(x)

        if self.use_residual:
            if self.shortcut:
                residual = self.shortcut(residual)
            
            if residual.shape != y.shape:
                diff_y = y.size(2) - residual.size(2)
                diff_x = y.size(3) - residual.size(3)

                residual = torch.nn.functional.pad(residual, 
                    [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])
            
            y = y + residual

        if self.activation == 'tanh':
            return torch.tanh(y)
        elif self.activation == 'relu':
            return torch.relu(y)
        return torch.nn.functional.leaky_relu(y, inplace=True)
    
class NoiseInjection(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        batch, _, height, width = x.shape
        noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise

class Module(torch.nn.Module):
    def __init__(self, 
                conv_blocks_number, 
                in_channels = 32, 
                internal_channels = 32, 
                out_channels = 32, 
                bypass = False, 
                max_pool = False,
                batch_norm = False,
                dropout = False
                ):
        super(Module, self).__init__()
        self.conv_in = ConvolutionalBlock(
                    in_channels = in_channels, 
                    out_channels = internal_channels,
                    # this was changed from False/bypass
                    bypass = False,
                    batch_norm = batch_norm
                )
        
        self.conv_blocks_number = conv_blocks_number
        if(self.conv_blocks_number != 0):
            self.blocks = torch.nn.Sequential(
                *[
                    ConvolutionalBlock(
                        in_channels = internal_channels, 
                        out_channels = internal_channels,
                        bypass = bypass,
                        batch_norm = batch_norm
                    ) for _ in range(conv_blocks_number)
                ]
            )

        self.conv_out = ConvolutionalBlock(
                    in_channels = internal_channels, 
                    out_channels = out_channels,
                    # this was set to always False
                    bypass = bypass,
                    batch_norm = batch_norm
                )
        
        self.max_pool = max_pool
        if(max_pool):
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = dropout
        if(dropout):
            self.dropout_layer = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv_in(x)
        if(self.conv_blocks_number != 0):
            x = self.blocks(x) 
        x = self.conv_out(x)
        if(self.max_pool):
            x = self.pool(x)
        if(self.dropout):
            x = self.dropout_layer(x)
        return x

class HeadBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, size = 32):
        super(HeadBlock, self).__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_channels * size * size, 1024),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace = True),

            torch.nn.Linear(256, 10)
        )
    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.head(x)
        return x
    
def evaluate_f1_score(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_spec, labels in tqdm.tqdm(test_loader):
            mel_spec = mel_spec.to(device)
            labels = labels.long().to(device)

            outputs = model(mel_spec)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    return f1