import torch
from torch import nn

from module.BaseBlocks import BasicConv2d


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class IDEM(nn.Module):
    def __init__(self, in_C, out_C):
        super(IDEM, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main1 = BasicConv2d(in_C,out_C,kernel_size=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        feat = self.fuse_down_mul(rgb + depth)
        return self.fuse_main(self.res_main(feat) + feat)





class Resudiual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resudiual, self).__init__()
        self.conv = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.sigmoid(x1)
        out = x1 * x
        return out


class Tdc3x3_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_1, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=1, padding=1)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_3, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_5, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=4, padding=4)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3,x4

class Tdc3x3_8(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_8, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=8, padding=8)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x4

class EDFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EDFM, self).__init__()
        self.one = Tdc3x3_1(in_channel, out_channel)
        self.two = Tdc3x3_3(in_channel, out_channel)
        self.three = Tdc3x3_5(in_channel, out_channel)
        self.four = Tdc3x3_8(in_channel,out_channel)
        self.fusion = BasicConv2d(out_channel , out_channel, 1)
        self.fusion1 = BasicConv2d(out_channel * 7, out_channel, 1)

    def forward(self, rgb, rgb_aux):
        x1, x2 = self.one(rgb_aux)
        x3, x4 = self.two(rgb_aux, x1)
        x5, x6 = self.three(rgb_aux, x3)
        x7 = self.four(rgb_aux,x5)
        x2 = x2 * rgb
        x4 = x4 * rgb
        x5 = x5 * rgb
        x2_1 = x2 - x4
        x4_1 = x4 - x6
        x6_1 = x6 - x7
        out = self.fusion1(torch.cat([x2,x4,x6,x7,x2_1,x4_1,x6_1],dim=1))
        out = self.fusion(torch.abs(out - rgb))
        return out

class BasicUpsample(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            nn.Conv2d(32,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)


class FDM(nn.Module):
    def __init__(self,):
        super(FDM, self).__init__()
        self.basicconv1 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=1)
        self.basicconv2 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=1)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,32,1),
            nn.ReLU()
        )
        self.basicconv3 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv4 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicupsample16 = BasicUpsample(scale_factor=16)
        self.basicupsample8 = BasicUpsample(scale_factor=8)
        self.basicupsample4 = BasicUpsample(scale_factor=4)
        self.basicupsample2 = BasicUpsample(scale_factor=2)
        self.basicupsample1 = BasicUpsample(scale_factor=1)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(160,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            )


    def forward(self,out_data_1,out_data_2,out_data_4,out_data_8,out_data_16):
        out_data_16 = self.basicconv1(out_data_16)
        out_data_16 = self.basicconv3(out_data_16)

        out_data_8 = self.basicconv1(out_data_8)
        out_data_8 = torch.cat([out_data_8,self.upsample1(out_data_16)],dim=1)
        out_data_8 = self.basicconv4(out_data_8)

        out_data_4 = self.basicconv1(out_data_4)
        out_data_4 = torch.cat([out_data_4,self.upsample1(out_data_8)],dim=1)
        out_data_4 = self.basicconv4(out_data_4)

        out_data_2 = self.basicconv2(out_data_2)
        out_data_2 = torch.cat([out_data_2,self.upsample1(out_data_4)],dim=1)
        out_data_2 = self.basicconv4(out_data_2)


        out_data_1 = self.basicconv2(out_data_1)
        out_data_1 = torch.cat([out_data_1,self.upsample1(out_data_2)],dim=1)
        out_data_1 = self.basicconv4(out_data_1)



        out_data_16 = self.basicupsample16(out_data_16)
        out_data_8 = self.basicupsample8(out_data_8)
        out_data_4 = self.basicupsample4(out_data_4)
        out_data_2 = self.basicupsample2(out_data_2)
        out_data_1 = self.basicupsample1(out_data_1)

        out_data = torch.cat([out_data_16,out_data_8,out_data_4,out_data_2,out_data_1],dim=1)

        out_data = self.reg_layer(out_data)


        return torch.abs(out_data)




