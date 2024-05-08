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

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(1, 65, 68096), patch_size=(65, 112),
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 ncc_weight:float=0.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, act_layer=nn.GELU, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * img_size[0], bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Contrastive specifics
        self.criterion = torch.nn.CosineSimilarity(dim=1)

        proj_dim = int(1024)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            # nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False)
            # nn.LayerNorm(embed_dim),
        )

        pred_dim = int(128)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            # nn.LayerNorm(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, embed_dim, bias=False),
            # nn.BatchNorm1d(embed_dim, affine=False)
            # nn.LayerNorm(embed_dim),
        )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.ncc_weight = ncc_weight

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, p*q*C)
        """
        p, q = self.patch_embed.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % q == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // q
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, q))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * q * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, p*q*C)
        imgs: (N, C, H, W)
        """
        p, q = self.patch_embed.patch_size
        h, w = self.patch_embed.grid_size
        assert h * w == x.shape[1]

        img_channels = int(x.shape[2] / (p*q))

        x = x.reshape(shape=(x.shape[0], h, w, p, q, img_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], img_channels, h * p, w * q))
        return imgs

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

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_all_patches(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        latent = self.norm(x)

        return latent

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*q*C]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs) # [N, L, p*q*C]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2

        # loss_patches = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss_patches = (loss_patches * mask).sum() / mask.sum()  # mean loss on removed patches

        # # REGULARIZATION (using masked patches)
        # loss_reg = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss_reg = (loss_reg * mask).sum() / mask.sum()  # mean loss on removed patches

        # # REGULARIZATION (using amplitude of the actual signal)
        # imgs_hat = self.unpatchify(pred)

        # imgs_hat_min = imgs_hat.min(dim=-1, keepdim=True)[0]
        # imgs_hat_max = imgs_hat.max(dim=-1, keepdim=True)[0]
        # imgs_min = imgs.min(dim=-1, keepdim=True)[0]
        # imgs_max = imgs.max(dim=-1, keepdim=True)[0]

        # loss_reg = (imgs_hat_min-imgs_min)**2 + (imgs_hat_max-imgs_max)**2 # penalizing difference in amplitude
        # loss_reg = loss_reg.mean()

        # REGULARIZATION (using normalized correlation coefficient of the actual signals)
        imgs_hat = self.unpatchify(pred)
        target_normalized = (imgs - imgs.mean(dim=-1, keepdim=True)) / (imgs.var(dim=-1, keepdim=True) + 1e-12)**0.5
        pred_normalized = (imgs_hat - imgs_hat.mean(dim=-1, keepdim=True)) / (imgs_hat.var(dim=-1, keepdim=True) + 1e-12)**0.5

        nb_of_signals = 1
        for dim in range(imgs.dim()-1): # all but the last dimension (which is the actual signal)
            nb_of_signals = nb_of_signals * imgs.shape[dim]

        cross_corrs = (1.0 / (imgs.shape[-1]-1)) * torch.sum(target_normalized * pred_normalized, dim=-1)
        ncc = cross_corrs.sum() / nb_of_signals

        loss = loss.sum() / (torch.numel(loss) + 1e-5)

        # return loss_patches
        # return (1-self.ncc_weight)*loss_patches + self.ncc_weight*(1-ncc)
        return (1-self.ncc_weight)*loss + self.ncc_weight*(1-ncc)


    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*q*C]
        loss = self.forward_loss(imgs, pred, mask)

        orig_patched = self.patchify(imgs)
        orig_masked_unpatched = self.unpatchify(orig_patched*(1-mask).unsqueeze(dim=-1))
        imgs_hat = self.unpatchify(pred)
        imgs_hat_masked = self.unpatchify(pred*(1-mask).unsqueeze(dim=-1))

        # contrastive part
        latent2, _, _ = self.forward_encoder(imgs, mask_ratio)

        z1 = latent[:, 1:, ...].mean(dim=1)     # global average pooling
        z2 = latent2[:, 1:, ...].mean(dim=1)    # global average pooling

        p1 = self.projector(z1)
        p2 = self.projector(z2)

        h1 = self.predictor(p1)
        h2 = self.predictor(p2)

        # loss_cos = - (self.criterion(h1, p2).mean() + self.criterion(h2, p1).mean()) * 0.5
        loss_cos = - (self.criterion(h1, p2.detach()).mean() + self.criterion(h2, p1.detach()).mean()) * 0.5
        # loss_cos = - (self.criterion(h1, z2).mean() + self.criterion(h2, z1).mean()) * 0.5
        # loss_cos = - (self.criterion(h1, z2.detach()).mean() + self.criterion(h2, z1.detach()).mean()) * 0.5

        # compare the similarity between the actual embeddings
        cos_embed = self.criterion(z1, z2).mean()

        # determine the std across all embeddings in the batch
        z_std = torch.nn.functional.normalize(z1, dim=-1).std(dim=0).mean() * z1.shape[-1]**0.5

        return loss, loss_cos, cos_embed, z_std, imgs_hat, imgs_hat_masked


def mae_vit_pluto_patchX_dec192d2b(**kwargs): # nb_params: 1.61M encoder, 0.37M decoder
    model = MaskedAutoencoderViT(
        embed_dim=256, depth=2, num_heads=8, # dim=32 per head
        decoder_embed_dim=160, decoder_depth=1, decoder_num_heads=8, # dim=20 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny_patchX_dec256d2b(**kwargs): # nb_params: 5.36M encoder, 1.7M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=3, num_heads=6, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny2_patchX_dec256d2b(**kwargs): # nb_params: 5.36M encoder, 1.7M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=3, num_heads=6, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tinyDeep_patchX_dec256d2b(**kwargs): # nb_params: 21.34M encoder, 1.8M decoder
    model = MaskedAutoencoderViT(
        embed_dim=192, depth=12, num_heads=3, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tinyDeep2_patchX_dec256d2b(**kwargs): # nb_params: 21.34M encoder, 1.8M decoder
    model = MaskedAutoencoderViT(
        embed_dim=192, depth=12, num_heads=3, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=512, depth=4, num_heads=8, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small2_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=512, depth=4, num_heads=8, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_smallDeep_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_smallDeep2_patchX_dec256d2b(**kwargs): # nb_params: 12.66M encoder, 1.74M decoder
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medium_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=640, depth=5, num_heads=10, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_medium2_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=640, depth=5, num_heads=10, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_mediumDeep_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=576, depth=12, num_heads=8, # dim=64 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_mediumDeep2_patchX_dec256d2b(**kwargs): # nb_params: 24.68M encoder, 1.77M decoder
    model = MaskedAutoencoderViT(
        embed_dim=576, depth=12, num_heads=8, # dim=64 per head
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patchX_dec256d4b(**kwargs): # nb_params: 53.91M encoder, 1.84M decoder
    model = MaskedAutoencoderViT(
        embed_dim=864, depth=6, num_heads=12, # dim=72 per head
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patchX_dec512d8b(**kwargs): # 86M params in total
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12, # dim=64 per head
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patchX_dec512d8b(**kwargs): # 307M params in total
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16, # dim=64 per head
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patchX_dec512d8b(**kwargs): # 632M params in total
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16, # dim=80 per head
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # dim=32 per head
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_pluto_patchX = mae_vit_pluto_patchX_dec192d2b  # decoder: 256 dim, 2 blocks

mae_vit_tiny_patchX = mae_vit_tiny_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_tiny2_patchX = mae_vit_tiny2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks
mae_vit_tinyDeep_patchX = mae_vit_tinyDeep_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_tinyDeep2_patchX = mae_vit_tinyDeep2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks

mae_vit_small_patchX = mae_vit_small_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_small2_patchX = mae_vit_small2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks
mae_vit_smallDeep_patchX = mae_vit_smallDeep_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_smallDeep2_patchX = mae_vit_smallDeep2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks

mae_vit_medium_patchX = mae_vit_medium_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_medium2_patchX = mae_vit_medium2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks
mae_vit_mediumDeep_patchX = mae_vit_mediumDeep_patchX_dec256d2b  # decoder: 256 dim, 2 blocks
mae_vit_mediumDeep2_patchX = mae_vit_mediumDeep2_patchX_dec256d2b  # decoder: 128 dim, 2 blocks

mae_vit_large_patchX = mae_vit_large_patchX_dec256d4b  # decoder: 256 dim, 2 blocks

mae_vit_base = mae_vit_base_patchX_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large = mae_vit_large_patchX_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge = mae_vit_huge_patchX_dec512d8b  # decoder: 512 dim, 8 blocks