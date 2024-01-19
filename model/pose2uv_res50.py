from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict
import torch
import torch.nn as nn
import numpy as np

from model.resnet import ResNet
from model.posenet import posenet
from model.segnet import segnet

# Specification
resnet_spec = {
    18: ([2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
    34: ([3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
    50: ([3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
    101: ([3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    152: ([3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
}


class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.LeakyReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.LeakyReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x

class Res_catconv(nn.Module):
    def __init__(self, backbone, head, generator):
        super(Res_catconv, self).__init__()
        self.generator = generator
        self.J_regressor_halpe = torch.from_numpy(np.load('data/J_regressor_halpe.npy')).cuda()

        self.segnet = segnet()
        self.posenet = posenet()
        self.backbone = backbone
        self.head = head
        self.conv3 = nn.Conv2d(in_channels=17, out_channels=17, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(in_channels=17, out_channels=17, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, data):
        
        img = data['img']
        fullheat = data['fullheat']

        pred = {}
        partialheat = self.posenet(img, fullheat[0])
        pre_mask = self.segnet(img, partialheat)
        mask = pre_mask[4]
        heat = partialheat[4]
        uv_inp = torch.cat([img, heat, mask], 1)
        uv_inp = self.conv3(uv_inp)
        uv_inp = self.conv4(uv_inp)
        latent = self.backbone(uv_inp)
        uv = self.head(latent)

        pred_verts = (self.generator.resample_t(uv) + 0.5) * 2
        pred_joints = torch.matmul(self.J_regressor_halpe, pred_verts)

        pred['latent'] = latent
        pred['pred_uv'] = uv
        pred['mask'] = mask
        pred['heatmap'] = heat
        pred['pred_mask'] = pre_mask
        pred['pred_heat'] = partialheat
        pred['pred_verts'] = pred_verts
        pred['pred_joints'] = pred_joints

        return pred

# Helper functions
def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 18
    config.num_deconv_layers = 8
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    config.depth_dim = 1
    config.input_channel = 3
    return config

# create network
def pose2uv_res50(generator, resnet_layers = 50):
    cfg = get_default_network_config()
    layers, channels, _ = resnet_spec[resnet_layers]
    backbone_net = ResNet(layers)
    head_net = DeconvHead(
        channels[-1], cfg.num_deconv_layers,
        cfg.num_deconv_filters, cfg.num_deconv_kernel,
        cfg.final_conv_kernel, 3, cfg.depth_dim
    )
    UV_net = Res_catconv(backbone_net, head_net, generator)
    return UV_net
