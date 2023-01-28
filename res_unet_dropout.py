import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Decoderblock(nn.Module):

    def __init__(self, in_channels, out_channels, up_kernel_size=3, up_stride=2, up_padding=1):
        super(Decoderblock, self).__init__()

        self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_kernel_size, stride=up_stride,
                                            padding=up_padding, output_padding=1)
        self.activation = nn.ReLU()
        # 
        self.first_block = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 
        self.second_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # dimensions commented for default parameters
        # x : (batch_size,in_channels,h,w), skip : (batch_size,out_channels,2h,2w)
        x = self.activation(self.upsampler(x))
        # concatenate with skip connection from encoder
        x = torch.cat([x, skip], dim=1)
        # x : (batch_size,2*out_channels,2h,2w)
        x = self.first_block(x)
        # x : (batch_size,out_channels,2h,2w)
        x = self.second_block(x)
        return x


class ResUnet(nn.Module):

    def __init__(self,num_classes=5,dprob=0.,pretrained=True):
        super(ResUnet, self).__init__()
        self.num_classes = num_classes
        self.resencoder = models.resnet34(pretrained=pretrained)

        # Divide encoder into first layer and subsequent layers
        # Subsequent layers match with decoder layers for skip connections
        self.first_filters = nn.Sequential(
            self.resencoder.conv1, self.resencoder.bn1,
            self.resencoder.relu
        )
        self.first_pool = self.resencoder.maxpool
        self.encoder_layers = nn.ModuleList([
            self.resencoder.layer1, self.resencoder.layer2,
            self.resencoder.layer3, self.resencoder.layer4])

        # middle layer
        self.middle_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # decoder layers
        self.decoder_layers = nn.ModuleList([
            Decoderblock(512, 256),
            Decoderblock(256, 128),
            Decoderblock(128, 64),
            Decoderblock(64, 64)
        ])
        # output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.dropout = nn.Dropout(dprob)

    def require_encoder_grad(self, requires_grad):
        # enable/disable encoder layer gradient updates

        blocks = [
            self.first_filters,
            *self.encoder_layers
        ]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):

        # to store intermediate encoder outputs for skip connections
        self.ints = []

        x = self.first_filters(x)
        self.ints.append(x)
        x = self.first_pool(x)

        for layer in self.encoder_layers:
            x = self.dropout(layer(x))
            self.ints.append(x)
        
        x = self.dropout(self.middle_layer(x))
        for inter, decoder_layer in zip(reversed(self.ints[:-1]), self.decoder_layers):
            x = self.dropout(decoder_layer(x, inter))
            
        del self.ints
        x = self.dropout(self.final_layer(x))
        
        return x
