# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, downstream_task='classification', 
                 masking_blockwise=False, mask_ratio=0.0, mask_c_ratio=0.0, mask_t_ratio=0.0, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.masking_blockwise = masking_blockwise
        self.mask_ratio = mask_ratio
        self.mask_c_ratio = mask_c_ratio
        self.mask_t_ratio = mask_t_ratio

        self.downstream_task = downstream_task

        self.global_pool = global_pool
        if self.global_pool == 'attention_pool':
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], batch_first=True)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_blockwise(self, x, mask_c_ratio, mask_t_ratio):
        """
        2D: ECG recording (N, 1, C, T) (masking c and t under mask_c_ratio and mask_t_ratio)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        C, T = int(self.img_size[-2] / self.patch_size[-2]), int(self.img_size[-1] / self.patch_size[-1])
        
        # mask C
        x = x.reshape(N, C, T, D)
        len_keep_C = int(C * (1 - mask_c_ratio))
        noise = torch.rand(N, C, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_C]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_C(C'), T, D

        # mask T
        x = x.permute(0, 2, 1, 3) # N C' T D => N T C' D
        len_keep_T = int(T * (1 - mask_t_ratio))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_C, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3) # N T' C' D => N C' T' D 
        
        x_masked = x_masked.reshape(N, len_keep_T*len_keep_C, D) # N C' T' D => N L' D
            
        return x_masked, None, None

    def forward_features(self, x):
        """
        x: [B=N, L, D], sequence
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]
        if self.masking_blockwise:
            x, _, _ = self.random_masking_blockwise(x, self.mask_c_ratio, self.mask_t_ratio)
        else:
            x, _, _ = self.random_masking(x, self.mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, 0, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool == 'attention_pool':
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, x_weights = self.attention_pool(q, k, v) # attention pool without cls token
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, :]
        x = self.fc_norm(x)
        
        if self.downstream_task == 'classification':
            return x if pre_logits else self.head(x)
        elif self.downstream_task == 'regression':
            return x if pre_logits else self.head(x) #.sigmoid()


def vit_pluto_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=256, depth=2, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=640, depth=6, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_big_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch200(**kwargs):
    model = VisionTransformer(
        patch_size=(65, 200), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch224(**kwargs):
    model = VisionTransformer(
        patch_size=(65, 224), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch112(**kwargs):
    model = VisionTransformer(
        patch_size=(65, 112), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patchX(**kwargs):
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model