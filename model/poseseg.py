from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn

from model.posenet import posenet
from model.segnet import segnet


class Res_catconv(nn.Module):
    def __init__(self, segnet, posenet):
        super(Res_catconv, self).__init__()
        self.segnet = segnet
        self.posenet = posenet

    def forward(self, data):
        img, fullheat = data['img'], data['fullheat']
        partialheat = self.posenet(img, fullheat[0])
        pre_mask = self.segnet(img, partialheat)

        return dict(encoded=None, decoded=None, mask=pre_mask[-1], heatmap=fullheat[-1], premask=pre_mask, preheat=partialheat)

# create network
def poseseg(generator=None):
    pose = posenet()
    seg = segnet()
    UV_net = Res_catconv(seg, pose)
    return UV_net
