import torch
import torch.nn as nn
import torch.nn.functional as F

class maskBasicBlock(nn.Module):

    def __init__(self, input_filters, num_filters, down_sampling=False):
        super(maskBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_filters, num_filters, 3, stride=(1 if down_sampling is False else 2), padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(input_filters, num_filters, 1, stride=(1 if down_sampling is False else 2))
        self.input_filters = input_filters
        self.num_filters = num_filters
        self.down_sampling = down_sampling

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if (self.num_filters != self.input_filters or (self.down_sampling is True)):
            identity = self.conv3(x)

        out = out + identity

        return self.relu(out)

class segnet(nn.Module):
    def __init__(self, in_channel=3):
        super(segnet, self).__init__()
        self.num_joints = 17
        self.conv0_0 = nn.Conv2d(in_channels=in_channel, out_channels=21, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv0_1 = nn.Conv2d(in_channels=in_channel, out_channels=21, kernel_size=3, stride=2, padding=2, dilation=2)
        self.conv0_2 = nn.Conv2d(in_channels=in_channel, out_channels=21, kernel_size=3, stride=2, padding=5, dilation=5)
        self.lrelu = nn.LeakyReLU(inplace=True) # Rectifier Nonlinearities Improve Neural Network Acoustic Models, ICML2013
        self.block1 = nn.Sequential(maskBasicBlock(63, 128, True), maskBasicBlock(128, 128))
        self.block2 = nn.Sequential(maskBasicBlock(128, 256, True), maskBasicBlock(256, 256))
        self.block3 = nn.Sequential(maskBasicBlock(256, 512, True), maskBasicBlock(512, 512))
        self.block4 = nn.Sequential(maskBasicBlock(512, 256), maskBasicBlock(256, 64))

        self.block5 = nn.Sequential(maskBasicBlock(512 + self.num_joints + 1, 256),maskBasicBlock(256, 64))
        self.block6 = nn.Sequential(maskBasicBlock(256 + self.num_joints + 1, 256),maskBasicBlock(256, 64))
        self.block7 = nn.Sequential(maskBasicBlock(128 + self.num_joints + 1, 128),maskBasicBlock(128, 64))
    
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.downsample = F.interpolate(x, scale_factor=0.5) # nn.Downsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax()

    def forward(self, x, fullheat):
        header = torch.cat([self.conv0_0(x),self.conv0_1(x),self.conv0_2(x)], 1)
        header = self.lrelu(header)
        block1 = self.block1(header)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        mask_0 = self.conv(self.block4(block3))
        mask_0 = self.lrelu(mask_0)

        stage0 = torch.cat([mask_0, fullheat[1], block3], 1)
        mask_1 = self.conv(self.block5(stage0))
        mask_1 = self.lrelu(mask_1)

        stage1 = torch.cat([self.upsample(mask_1), fullheat[2], block2], 1) ### only use full heatmap in first stage
        mask_2 = self.conv(self.block6(stage1))
        mask_2 = self.lrelu(mask_2)

        stage2 = torch.cat([self.upsample(mask_2), fullheat[3], block1], 1)
        mask_3 = self.conv(self.block7(stage2))
        mask_3 = self.lrelu(mask_3)

        mask_out = self.upsample(mask_3)
        mask_out = self.upsample(mask_out)

        return [mask_0, mask_1, mask_2, mask_3, mask_out]


# import torch
# import torch.nn as nn

# class maskBasicBlock(nn.Module):

#     def __init__(self, input_filters, num_filters, down_sampling=False):
#         super(maskBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(input_filters, num_filters, 3, stride=(1 if down_sampling is False else 2), padding=1)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
#         self.conv3 = nn.Conv2d(input_filters, num_filters, 1, stride=(1 if down_sampling is False else 2))
#         self.input_filters = input_filters
#         self.num_filters = num_filters
#         self.down_sampling = down_sampling

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if (self.num_filters != self.input_filters or (self.down_sampling is True)):
#             identity = self.conv3(x)

#         out = out + identity

#         return self.relu(out)

# class SegNet(nn.Module):
#     def __init__(self, in_channel=3):
#         super(SegNet, self).__init__()
#         self.conv0_0 = nn.Conv2d(in_channels=in_channel, out_channels=21, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.conv0_1 = nn.Conv2d(in_channels=in_channel, out_channels=21, kernel_size=3, stride=2, padding=2, dilation=2)
#         self.conv0_2 = nn.Conv2d(in_channels=in_channel, out_channels=21, kernel_size=3, stride=2, padding=5, dilation=5)
#         self.lrelu = nn.LeakyReLU(inplace=True) # Rectifier Nonlinearities Improve Neural Network Acoustic Models, ICML2013
#         self.block1 = nn.Sequential(maskBasicBlock(63, 128, True),maskBasicBlock(128, 128))
#         self.block2 = nn.Sequential(maskBasicBlock(128, 256, True),maskBasicBlock(256, 256))
#         self.block3 = nn.Sequential(maskBasicBlock(256, 512, True),maskBasicBlock(512, 512))
#         self.block4 = nn.Sequential(maskBasicBlock(512, 256),maskBasicBlock(256, 64))
#         self.block5 = nn.Sequential(maskBasicBlock(513, 256),maskBasicBlock(256, 64))
#         self.block6 = nn.Sequential(maskBasicBlock(257, 256),maskBasicBlock(256, 64))
#         self.block7 = nn.Sequential(maskBasicBlock(129, 128),maskBasicBlock(128, 64))
#         self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         header = torch.cat([self.conv0_0(x),self.conv0_1(x),self.conv0_2(x)], 1)
#         header = self.lrelu(header)
#         block1 = self.block1(header)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         heat_map0 = self.conv(self.block4(block3))
#         heat_map0 = self.lrelu(heat_map0)

#         stage0 = torch.cat([heat_map0, block3], 1)
#         heat_map1 = self.conv(self.block5(stage0))
#         heat_map1 = self.lrelu(heat_map1)

#         stage1 = torch.cat([self.upsample(heat_map1), block2], 1)
#         heat_map2 = self.conv(self.block6(stage1))
#         heat_map2 = self.lrelu(heat_map2)

#         stage2 = torch.cat([self.upsample(heat_map2), block1], 1)
#         heat_map3 = self.conv(self.block7(stage2))
#         heat_map3 = self.lrelu(heat_map3)

#         mask = self.upsample(heat_map3)
#         mask = self.upsample(mask)
#         # mask = self.softmax(mask)

#         return (heat_map0, heat_map1, heat_map2, heat_map3, mask)



# class segnet(nn.Module):
#     def __init__(self):
#         super(segnet, self).__init__()
#         self.segnet = SegNet(in_channel=4)
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1)

#     def forward(self, x, img):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         hm0, hm1, hm2, hm3, mask = self.segnet(x)
#         return [hm0, hm1, hm2, hm3, mask]









