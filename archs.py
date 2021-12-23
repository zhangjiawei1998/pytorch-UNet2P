import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['UNet', 'NestedUNet', 'UNet2P']

# 源代码作者搭建的UNet和UNet++
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, *sn):
        
        for i in range(len(sn)):
            #因卷积操作会导致特征图尺寸不对应，通过F.pad补成一致
            diffH = sn[i].shape[2] - x.shape[2]
            diffW = sn[i].shape[3] - x.shape[3]
            p2d = (diffW // 2, diffW - diffW // 2, 
                   diffH // 2, diffH - diffH // 2)
            x = F.pad(x, p2d, mode='reflect')
            x = torch.cat([x, sn[i]], dim=1)#通过torch.cat函数将channels维度加起来
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0), x0_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0), x1_0 )
        x0_2 = self.conv0_2(self.up(x1_1), x0_0, x0_1)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0), x2_0)
        x1_2 = self.conv1_2(self.up(x2_1), x1_0, x1_1)
        x0_3 = self.conv0_3(self.up(x1_2), x0_0, x0_1, x0_2)

        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x3_1 = self.conv3_1(self.up(x4_0), x3_0)
        x2_2 = self.conv2_2(self.up(x3_1), x2_0, x2_1)
        x1_3 = self.conv1_3(self.up(x2_2), x1_0, x1_1, x1_2)
        x0_4 = self.conv0_4(self.up(x1_3), x0_0, x0_1, x0_2, x0_3)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

# 2021/12/23 zjw搭建的UNet++，没有测试过
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1 ,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),#进行覆盖运算, 节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1 ,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)        
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=False, n_concat=2):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#不会对channels进行改变
            self.conv = DoubleConv(in_channels + (n_concat-2)*out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)#会对channels进行改变
            self.conv = DoubleConv(in_channels + (n_concat-2)*out_channels, out_channels)
        
    #x : input
    #sn: skip connect
    #反卷积之后，和skip connect过来的channel维度相加
    def forward(self, x, *sc):
        
        x = self.up(x)
        
        for i in range(len(sc)):
            #因卷积操作会导致特征图尺寸不对应，通过F.pad补成一致
            diffH = sc[i].shape[2] - x.shape[2]
            diffW = sc[i].shape[3] - x.shape[3]
            p2d = (diffW // 2, diffW - diffW // 2, 
                   diffH // 2, diffH - diffH // 2)
            x = F.pad(x, p2d, mode='reflect')
            x = torch.cat([x, sc[i]], dim=1)#通过torch.cat函数将channels维度加起来
        # print('x.shape=', x.shape)
        # print(f'[in_channels:{self.conv.in_channels},out_channels:{self.conv.out_channels} ]')
        # input is CHW
        return self.conv(x)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

class UNet2P(nn.Module):
    def __init__(self, n_classes, in_channels=3, deep_supervision=False):
        super().__init__()
        
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        
        # Down-sampling
        self.X_00 = DoubleConv(in_channels, 64)
        self.X_10 = Down(64, 128)
        self.X_20 = Down(128, 256)
        self.X_30 = Down(256, 512)
        self.X_40 = Down(512, 1024)
        
        # Up-sampling
        # 网络结构图见edge集锦，n_concat随着深度增加增大的原因：
        # skip connection的存在会使得前向神经元的输出加到之后的
        # 神经元，通过增加n_concat参数改变unetConv2中的输入通道
        # 数(skip connect + input)  2021/12/08 notzjw
        self.X_01 = Up(128, 64)
        self.X_11 = Up(256, 128)
        self.X_21 = Up(512, 256)
        self.X_31 = Up(1024, 512)
        
        self.X_02 = Up(128, 64,  n_concat = 3)
        self.X_12 = Up(256, 128, n_concat = 3)
        self.X_22 = Up(512, 256, n_concat = 3)
        
        self.X_03 = Up(128, 64,  n_concat = 4)
        self.X_13 = Up(256, 128, n_concat = 4)
        
        self.X_04 = Up(128, 64,  n_concat = 5)

        # final conv
        # 对每个channels维度上进行全连接，即通道融合
        self.final_1 = nn.Conv2d(64, n_classes, 1)
        self.final_2 = nn.Conv2d(64, n_classes, 1)
        self.final_3 = nn.Conv2d(64, n_classes, 1)
        self.final_4 = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, input):
        
        # Down-sampling
        X_00 = self.X_00(input)
        X_10 = self.X_10(X_00)
        X_20 = self.X_20(X_10)
        X_30 = self.X_30(X_20)
        X_40 = self.X_40(X_30)
        
        # Up-sampling
        X_01 = self.X_01(X_10, X_00)
        X_11 = self.X_11(X_20, X_10)
        X_21 = self.X_21(X_30, X_20)
        X_31 = self.X_31(X_40, X_30)
        
        # Up-sampling
        X_02 = self.X_02(X_11, X_00, X_01)
        X_12 = self.X_12(X_21, X_10, X_11)
        X_22 = self.X_22(X_31, X_20, X_21)
        
        # Up-sampling
        X_03 = self.X_03(X_12, X_00, X_01, X_02)
        X_13 = self.X_13(X_22, X_10, X_11, X_12)
        
        # Up-sampling
        X_04 = self.X_04(X_13, X_00, X_01, X_02, X_03)
        
        # Final conv
        output1 = self.final_1(X_01)
        output2 = self.final_2(X_02)
        output3 = self.final_3(X_03)
        output4 = self.final_4(X_04)
        
        if self.deep_supervision:
            return [output1, output2, output3, output4]
        else:
            return (output1 + output2 + output3 + output4) / 4
        
        
        
        