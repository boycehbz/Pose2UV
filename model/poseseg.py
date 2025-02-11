from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn

from model.posenet import posenet
from model.segnet import segnet


class poseseg(nn.Module):
    def __init__(self, generator=None, pretrain_poseseg=False, use_full_heat=True):
        super(poseseg, self).__init__()
        self.use_full_heat = use_full_heat
        self.segnet = segnet()
        self.posenet = posenet(use_full_heat=use_full_heat)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, data):
        img = data['img']
        if self.use_full_heat:
            full_heat = data['input_heat'][0]
        else:
            full_heat = None

        partialheat = self.posenet(img, full_heat=full_heat)
        pre_mask = self.segnet(img, partialheat)

        input_heat = data['input_heat'][-1]

        return dict(mask=pre_mask[-1], heatmap=input_heat, premask=pre_mask, preheat=partialheat)

