from collections import OrderedDict

import torch
import torch.nn as nn

def getMaskDeleterModel():
    return MaskDeleterModel()

class MaskDeleterModel(nn.Module):
    def __init__(self, unet:nn.Module=None):
        super(MaskDeleterModel, self).__init__()
        
        self.unet = unet if unet != None else UNet()
        self.isPreTrainedUnet = (unet != None)
        
        # Untrained UNet : Avg loss per evaluation picture -> 54.26 on epoch 50
        
        """V1, pre-trained Unet -> Avg loss per evaluation picture : 49.73 after epoch 50
        self.deleter = torch.nn.Sequential( 
            torch.nn.Conv2d(4, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(8, 16, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(16, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(8, 3, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
        )
        """
        
        """ Not great
        #V2
        self.deleter = torch.nn.Sequential( 
            torch.nn.Conv2d(4, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.Conv2d(16, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            torch.nn.ReLU(), #
            torch.nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(8, 3, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU() #
        )
        """
        
        """
        #V3 -> Avg loss per evaluation picture : 33.53
        self.deleter = torch.nn.Sequential( 
            torch.nn.Conv2d(4, 8, 7, 1, padding=3, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(8, 16, 7, 1, padding=3, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(16, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(8, 8, 5, 1, padding=2, padding_mode="replicate", bias=False),
            torch.nn.ReLU(), #
            torch.nn.Conv2d(8, 3, 3, 1, padding=1, padding_mode="replicate", bias=False),
            torch.nn.ReLU(),
        )
        """
        
        #V4
        self.deleter = Deleter()
        
    def forward(self, x):
        if (self.isPreTrainedUnet):
            with torch.no_grad():
                writingMask = self.unet(x)
        else:
            writingMask = self.unet(x)
        
        result = self.deleter(torch.concat((x,writingMask),dim=1))
        
        return result

    
class Deleter(nn.Module):

    #def __init__(self, in_channels=4, out_channels=3, init_features=4): #V4 -> Avg loss per evaluation picture : 35.64 at epoch 50 (But a bit cleaner than V3)
    #def __init__(self, in_channels=4, out_channels=3, init_features=8): #V5 -> Avg loss per evaluation picture : 27.36 at epoch 30 (cleaner than V4)
    def __init__(self, in_channels=4, out_channels=3, init_features=16): #V6 -> Avg loss per evaluation picture : 12.45 at epoch 50 (clean AF)
        super(Deleter, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=init_features*2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=init_features*2),
            nn.ReLU(), #
            nn.Conv2d(in_channels=init_features*2, out_channels=init_features*2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=init_features*2),
            nn.ReLU() #
        )
        self.max1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=init_features*2, out_channels=init_features*3, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(num_features=init_features*3),
            nn.ReLU(), #
            nn.Conv2d(in_channels=init_features*3, out_channels=init_features*3, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(num_features=init_features*3),
            nn.ReLU() #
        )
        self.max2 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=init_features*3, out_channels=init_features*4, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(num_features=init_features*4),
            nn.ReLU(), #
            nn.Conv2d(in_channels=init_features*4, out_channels=init_features*4, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(num_features=init_features*4),
            nn.ReLU() #
        )
        
        self.upconv1 = nn.ConvTranspose2d(init_features*4, init_features*3, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=(init_features*3)*2, out_channels=init_features*3, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=init_features*3),
            nn.ReLU(), #
            nn.Conv2d(in_channels=init_features*3, out_channels=init_features*3, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=init_features*3),
            nn.ReLU() #
        )
        self.upconv2 = nn.ConvTranspose2d(init_features*3, init_features*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=(init_features*2)*2, out_channels=init_features*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=init_features*2),
            nn.ReLU(), #
            nn.Conv2d(in_channels=init_features*2, out_channels=init_features*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=init_features*2),
            nn.ReLU() #
        )
        
        self.conv = nn.Conv2d(in_channels=init_features*2, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.max1(enc1))

        bottleneck = self.bottleneck(self.max2(enc2))

        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.dec1(dec1)

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)

        return torch.nn.functional.relu(self.conv(dec2))
            
    
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )