# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
# import timm
import math
from model.tokenpose_base import TokenPose_TB_base
from model.hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

from model.config import cfg
from model.config import update_config

class TokenPose_B(nn.Module):

    def __init__(self, cfg):

        super(TokenPose_B, self).__init__()

        ##################################################
        self.pre_feature = HRNET_base(cfg)
        self.transformer = TokenPose_TB_base(feature_size=[cfg.MODEL.IMAGE_SIZE[1]//4,cfg.MODEL.IMAGE_SIZE[0]//4],patch_size=[cfg.MODEL.PATCH_SIZE[1],cfg.MODEL.PATCH_SIZE[0]],
                                 num_keypoints = cfg.MODEL.NUM_JOINTS,dim =cfg.MODEL.DIM,
                                 channels=cfg.MODEL.BASE_CHANNEL,
                                 depth=cfg.MODEL.TRANSFORMER_DEPTH,heads=cfg.MODEL.TRANSFORMER_HEADS,
                                 mlp_dim = cfg.MODEL.DIM*cfg.MODEL.TRANSFORMER_MLP_RATIO,
                                 apply_init=cfg.MODEL.INIT,
                                 hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0]//8,
                                 heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0],
                                 heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1],cfg.MODEL.HEATMAP_SIZE[0]],
                                 pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE)
        ###################################################3
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, data):
        x = data['img']
        x = self.pre_feature(x)
        x = self.transformer(x)

        heatmap = self.upsample(x)
        heatmap = self.upsample(heatmap)

        pred_heat = [None, None, None, x, heatmap]
        pred_mask = [None, None, None, None, data['mask'][-1]]

        return dict(mask=data['mask'][-1], heatmap=data['gt_heat'][-1], premask=pred_mask, preheat=pred_heat)

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def pose_tokenpose_b(generator):
    update_config(cfg, 'model/config/tokenpose_b_256_192_patch43_dim192_depth12_heads8.yaml')
    model = TokenPose_B(cfg)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
