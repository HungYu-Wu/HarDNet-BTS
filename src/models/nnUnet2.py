"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from .layers import ConvBnRelu, UBlock1, UBlock, conv1x1, UBlockCbam, CBAM


class Unet2(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self,deep_supervision=False,norm_layer=None,**kwargs):
        super(Unet2, self).__init__()
        #features = [width * 2 ** i for i in range(5)]
        #print(features)
        
        dropout=0
        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock1(4, 64, 64, norm_layer, dropout=dropout)
        self.encoder2 = UBlock(64, 128, 128, norm_layer, dropout=dropout)
        self.encoder3 = UBlock(128, 256, 256, norm_layer, dropout=dropout)
        self.encoder4 = UBlock(256, 512, 512, norm_layer, dropout=dropout)
        self.encoder5 = UBlock(512, 512, 512, norm_layer, dropout=dropout)
        self.encoder6 = UBlock(512, 512, 512, norm_layer, dropout=dropout)
        
        self.trans1 = nn.ConvTranspose3d(512,512,kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.trans2 = nn.ConvTranspose3d(256,512,kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.trans3 = nn.ConvTranspose3d(256,256,kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.trans4 = nn.ConvTranspose3d(128,128,kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.trans5 = nn.ConvTranspose3d(64,64,kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        
        
        #self.bottom = UBlock(features[4], features[4], features[4], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = UBlock1(1024, 512, 256, norm_layer, dropout=dropout)

        #self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock1(1024, 512, 256, norm_layer, dropout=dropout)
        self.decoder2 = UBlock1(512, 256, 128, norm_layer, dropout=dropout)
        self.decoder1 = UBlock1(256, 128, 64, norm_layer, dropout=dropout)
        self.decoder0 = UBlock1(128, 64, 32, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(32, 3)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(256, 3),
                nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(128, 3),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(64, 3),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(32, 3),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        down1 = self.encoder1(x)
        #down2 = self.downsample(down1)
        down2 = self.encoder2(down1)
        #down3 = self.downsample(down2)
        down3 = self.encoder3(down2)
        #down4 = self.downsample(down3)
        down4 = self.encoder4(down3)
        down5 = self.encoder5(down4)
        down6 = self.encoder6(down5)
        
        # Decoder
        bottom = self.trans1(down6)
        
        #bottom = self.bottom(down4)
        #print(down5.shape, bottom.shape)
        bottom_2 = self.bottom_2(torch.cat([down5, bottom], dim=1))
        #print(bottom_2.shape)
        

        up3 = self.trans2(bottom_2)
        #print(up3.shape, down4.shape)
        up3 = self.decoder3(torch.cat([down4, up3], dim=1))
        up2 = self.trans3(up3)
        up2 = self.decoder2(torch.cat([down3, up2], dim=1))
        up1 = self.trans4(up2)
        up1 = self.decoder1(torch.cat([down2, up1], dim=1))
        up0 = self.trans5(up1)
        up0 = self.decoder0(torch.cat([down1, up0], dim=1))

        out = self.outconv(up0)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [up3, up2, up1, up0],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out



