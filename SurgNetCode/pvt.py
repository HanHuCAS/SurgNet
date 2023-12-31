# --------------------------------------------------------
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
#from curses import window
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp, Block

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

import os, sys
sys.path.append('..')

from models_mae_pvt import PatchEmbed, PVTBlock, PatchMerge

@BACKBONES.register_module()
class PVT(nn.Module):
    """ Pyramid Vision Transformer 
    """
    def __init__(self, img_size=224, num_classes=80, patch_size=4, in_chans=3, 
                 embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm, out_indices=[2, 6, 12, 15], fpn_out_dim=768, is_fpn_out_layer = False): 
        super().__init__()
        self.patch_size = patch_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dims = embed_dims  # num_features for consistency with other models

        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches

        self.num_layers = len(depths)

        # during finetuning we let the pos_embed learn
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]), requires_grad=True)  

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        idx = 0
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                layer = PVTBlock(dim=embed_dims[i_layer], 
                                 num_heads=num_heads[i_layer],
                                 sr_ratio=sr_ratios[i_layer],
                                 mlp_ratio=mlp_ratios[i_layer],
                                 qkv_bias=True, qk_scale=None,
                                 drop_path=dpr[idx],
                                 downsample=PatchMerge(
                                     patch_size=2, 
                                     in_chans=embed_dims[i_layer - 1], 
                                     embed_dim=embed_dims[i_layer]
                                 ) if downsample_flag else None
                )
                self.blocks.append(layer)
                idx += 1

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
            
        if is_fpn_out_layer == True:
            self.fpn1 = nn.Conv2d(512, fpn_out_dim, 1)
            self.fpn2 = nn.Conv2d(512, fpn_out_dim, 1)
            self.fpn3 = nn.Conv2d(512, fpn_out_dim, 1)
            self.fpn4 = nn.Conv2d(512, fpn_out_dim, 1)
        self.is_fpn_out_layer = is_fpn_out_layer
		
        # --------------------------------------------------------------------------
        self.norm_box = nn.ModuleList()
        self.norm_box.append(norm_layer(embed_dims[0]))
        self.norm_box.append(norm_layer(embed_dims[1]))
        self.norm_box.append(norm_layer(embed_dims[2]))
        self.norm_box.append(norm_layer(embed_dims[3]))

        self.decoder_seg0 = nn.ModuleList()
        #self.decoder_seg1 = nn.ModuleList()
        out_channel = 512
        self.max_segmentation_class_number = 256
        self.num_prototype = 4
        
        self.decoder_seg0.append(nn.Sequential(nn.Conv2d(embed_dims[0], out_channel, 1), nn.GELU()))
        self.decoder_seg0.append(nn.Sequential(nn.Conv2d(embed_dims[1], out_channel, 1), nn.GELU()))
        self.decoder_seg0.append(nn.Sequential(nn.Conv2d(embed_dims[2], out_channel, 1), nn.GELU()))
        self.decoder_seg0.append(nn.Sequential(nn.Conv2d(embed_dims[3], out_channel, 1), nn.GELU()))

        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        
        self.decoder_predseg_0 = nn.Sequential(nn.Conv2d(512,512,kernel_size=1,stride=1, padding=0), nn.GELU())
        self.decoder_predseg_1 = nn.Sequential(nn.Linear(4096,4096, bias=True), nn.GELU())
        
        self.decoder_predreg = nn.Sequential(nn.Conv2d(32,32,kernel_size=1,stride=1, padding=0), nn.GELU(),nn.Conv2d(32,3,kernel_size=1,stride=1, padding=0))
        self.decoder_pred_prototype = nn.Sequential(nn.Conv2d(32,self.num_prototype,kernel_size=1,stride=1, padding=0))
        
        self.decoder_pred_label0 = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(32,self.num_prototype,kernel_size=1,stride=1, padding=0), nn.GELU())
        self.decoder_pred_label1 = nn.Sequential(nn.Linear(4096,self.max_segmentation_class_number, bias=True))
        #
        #loss
        self.down = nn.Upsample(scale_factor = 0.5)
		
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward_features(self, x):
        B, _, H, W = x.size()
        x = self.patch_embed(x)
        H, W = H//self.patch_size, W//self.patch_size

        # add position embedding
        x = x + self.pos_embed
		
        out_indices=[2, 6, 12, 15]
        features = []
        """for i, blk in enumerate(self.blocks):
            x, (H, W) = blk(x, H, W)
            if i in self.out_indices:
                #xp = x.permute(0, 2, 1).reshape(B, -1, H, W)
                #features.append(xp.contiguous())
                features.append((x, H, W))"""
        k = 0
        for i, blk in enumerate(self.blocks):
            x, (H, W) = blk(x, H, W)
            if i in out_indices:
                xp = self.norm_box[k](x).permute(0, 2, 1).reshape(B, -1, H, W)
                k += 1
                features.append(xp.contiguous())


        # build laterals
        """out_up = []
        for i in range(4):
            H = features[i][1]
            W = features[i][2]
            #fpn = self.decoder_seg1[i](features[i][0].permute(0, 2, 1))
            fpn = (features[i][0].permute(0, 2, 1))
            fpn = self.decoder_seg0[i](fpn.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
            out_up.append(fpn)"""
			

        out_up = []
        # build laterals
        for i in range(4):
            out_up.append(self.decoder_seg0[i](features[i]))


        # build top-down path
        for i in range(4 - 1, 0, -1):
            out_up[i-1] += self.upsample(out_up[i])
        features = out_up

        if self.is_fpn_out_layer == True:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x
