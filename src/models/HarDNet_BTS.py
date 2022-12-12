import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections


class SoftPooling3D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling3D, self).__init__()
        self.avgpool = torch.nn.AvgPool3d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm3d(out_channels))
        self.add_module('relu', Mish())

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)


    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = Mish()
        #nn.ReLU(inplace=True)
        

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3), skip.size(4)),
                mode="trilinear",
                align_corners=True,
                            )
        if concat:                            
          out = torch.cat([out, skip], 1)
          
        return out

class hardnet4(nn.Module):
    def __init__(self, n_classes=3, deep_supervision=True):
        super(hardnet4, self).__init__()
        self.deep_supervision = deep_supervision
        #'''
        first_ch  = [16,32,32,64]
        ch_list = [    128, 256, 320, 320, 1024]
        grmul = 1.7
        gr       = [    14, 16, 20, 20,160]
        n_layers = [     8, 16, 16 ,16]#,  4]
        
        blks = len(n_layers) 
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append (ConvLayer(in_channels=4, out_channels=first_ch[2], kernel=3,stride=2) )
        self.base.append ( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )
        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append ( blk )
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1)

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            #######################################
            if i < blks-1:      
              self.base.append ( SoftPooling3D(kernel_size=2, strides=2) )
              #self.base.append ( nn.AvgPool3d(kernel_size=2, stride=2) )  

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks
        
        ######################
        #deep_sup
        ######################
        #'''
        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                nn.Conv3d(320, n_classes, kernel_size=1, stride=1, bias=True),
                nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True))


            self.deep_bottom2 = nn.Sequential(
                nn.Conv3d(328, n_classes, kernel_size=1, stride=1, bias=True),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))
               

            self.deep3 = nn.Sequential(
                nn.Conv3d(262, n_classes, kernel_size=1, stride=1, bias=True),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))
               

            self.deep2 = nn.Sequential(
                nn.Conv3d(124, n_classes, kernel_size=1, stride=1, bias=True),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))
                
        #'''
        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        
        for i in range(n_blocks-1,-1,-1):
            ##################################
                
                self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
                self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))
                cur_channels_count = cur_channels_count//2

                blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            
                self.denseBlocksUp.append(blk)
                prev_block_channels = blk.get_out_ch()
                cur_channels_count = prev_block_channels


        self.finalConv = nn.Conv3d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
               padding=0, bias=True)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        ##################################
        #skip = skip_connections.pop()  
        ##################################
        for i in range(self.n_blocks):
            
            skip = skip_connections.pop()
            if i ==0:
                x4 = out;
            
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
            if i ==0:
                x3=  out;
            elif i ==1:
                x2 = out;
            elif i ==2:
                x1 = out;
        
        out = self.finalConv(out)
        out = F.interpolate(
                            out,
                            size=(size_in[2], size_in[3],size_in[4]),
                            mode="trilinear",
                            align_corners=True)
           
        #####################
        
        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [x4, x3, x2, x1],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
                
            return out, deeps
        
        
        
        return out 


# +
#'''
import time

if __name__ == "__main__":
    model = hardnet4()
    total_params = sum(p.numel() for p in model.parameters())
    #print('Parameters: ', total_params )
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    model.to(device)
    total_time = 0
    start_time = 0
    time_all = 0    
    #images = tor
    print(model)
    for i in range(100):
        images = torch.randn((1, 4, 128, 128,128)).to("cuda")
       
        if i == 0:
            with torch.no_grad():
                 output = model(images)
        else:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                  outputs = model(images)
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time


            print(
                "Inference time \
                  (iter {0:5d}):  {1:3.5f} fps".format(
                    i + 1, 1 / elapsed_time
                )
            )
            total_time += 1/elapsed_time
    print(total_time/100)
#'''
# -


