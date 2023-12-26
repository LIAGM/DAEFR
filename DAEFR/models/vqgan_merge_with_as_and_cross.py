import os, math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import pytorch_lightning as pl
from main import instantiate_from_config

from DAEFR.modules.vqvae.utils import get_roi_regions

from DAEFR.modules.vqvae.vqvae_arch import MultiHeadAttnBlock

# codeformer code
def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt
# codeformer code ------------------------




def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DAEFRModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 transformer_config=None,
                 cond_stage_config=None,
                 permuter_config=None,
                 ckpt_path=None,
                 ckpt_path_HQ=None,
                 ckpt_path_LQ=None,
                 encoder_codebook_type=None,
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000]
                 ):
        super().__init__()

        # import pdb
        # pdb.set_trace()

        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig)

        # code transformer
        # if transformer_config is not None:
        #     self.transformer = instantiate_from_config(config=transformer_config)
        # if cond_stage_config is not None:
        #     self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "DAEFR.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)

        lossconfig['params']['distill_param'] = ddconfig['params']
        # self.loss = instantiate_from_config(lossconfig)
        # get the weights from HQ and LQ checkpoints
        if (ckpt_path_HQ is not None) and (ckpt_path_LQ is not None):
            self.init_from_ckpt_two(
                ckpt_path_HQ, ckpt_path_LQ, ignore_keys=ignore_keys)
        elif ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']

        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step

        self.encoder_codebook_type = encoder_codebook_type

        # self.linear_projection = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        # )

        self.cross_attention = MultiHeadAttnBlock(in_channels=256,head_size=8)

        # codeformer code-----------------------------------
        dim_embd=512
        n_head=8
        n_layers=9 
        codebook_size=1024
        latent_size=256
        connect_list=['32', '64', '128', '256']
        fix_modules=['quantize','generator']
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2
        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# codeformer code-----------------------------------

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())

        print('In single checkpoint load function-----------')
        # import pdb
        # pdb.set_trace()

        # HQLQ encoder + HQ decoder
        HQ_ignore_keys = ['vqvae.encoder',
                          'vqvae.quant_conv.weight',
                          'vqvae.quant_conv.bias']
        LQ_ignore_keys = ['loss_LQ',
                          'vqvae_LQ.quantize.embedding.weight',
                          'vqvae_LQ.post_quant_conv.weight',
                          'vqvae_LQ.post_quant_conv.bias',
                          'vqvae_LQ.decoder']

        print("All keys before delete = ", len(keys))
        # From checkpoint all: 921
        # 'vqvae.encoder': 170
        # 'vqvae.decoder': 230
        # 'loss': 55
        HQ_count = {'vqvae.encoder': 0,
                    'vqvae.HQ_encoder': 0,
                    'vqvae.decoder': 0,
                    'loss': 0,
                    'vqvae.quant_conv.weight': 0,
                    'vqvae.quant_conv.bias': 0,
                    'vqvae.HQ_quant_conv.weight': 0,
                    'vqvae.HQ_quant_conv.bias': 0,
                    'vqvae.quantize.embedding.weight': 0,
                    'vqvae.post_quant_conv.weight': 0,
                    'vqvae.post_quant_conv.bias': 0}
        LQ_count = {'vqvae_LQ.encoder': 0,
                    'vqvae_LQ.decoder': 0,
                    'loss_LQ': 0,
                    'vqvae_LQ.quant_conv.weight': 0,
                    'vqvae_LQ.quant_conv.bias': 0,
                    'vqvae_LQ.quantize.embedding.weight': 0,
                    'vqvae_LQ.LQ_quantize.embedding.weight': 0,
                    'vqvae_LQ.post_quant_conv.weight': 0,
                    'vqvae_LQ.post_quant_conv.bias': 0}

        # delete and replace keys in HQ Keys
        for k in keys:
            for ik in HQ_ignore_keys:
                if k.startswith(ik):
                    HQ_count[ik] = HQ_count[ik] + 1
                    HQ_count[ik.replace('vqvae.', 'vqvae.HQ_')] = HQ_count[ik.replace('vqvae.', 'vqvae.HQ_')] + 1
                    # print("Deleting key {} from state_dict.".format(k))
                    sd[k.replace('vqvae.', 'vqvae.HQ_')] = sd[k]
                    del sd[k]
       
        
        delete_LQ_keys = 0
        # delete and replace keys in LQ Keys
        for k in keys:
            for ik in LQ_ignore_keys:
                if (k == 'vqvae_LQ.quantize.embedding.weight') and k.startswith(ik):
                    LQ_count['vqvae_LQ.quantize.embedding.weight'] = LQ_count['vqvae_LQ.quantize.embedding.weight'] + 1
                    LQ_count['vqvae_LQ.LQ_quantize.embedding.weight'] = LQ_count['vqvae_LQ.LQ_quantize.embedding.weight'] + 1
                    sd[k.replace('vqvae_LQ.', 'vqvae.LQ_')] = sd[k]
                    del sd[k]
                elif k.startswith(ik):
                    LQ_count[ik] = LQ_count[ik] + 1
                    delete_LQ_keys = delete_LQ_keys + 1
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        keys = list(sd.keys())
        print('key number before rename = ',len(keys))
        
        print('Delete LQ keys number = ', delete_LQ_keys)
        # Remain keys---------
        # HQ keys = 460
        # LQ keys = 173
        
        # import pdb
        # pdb.set_trace()

        for k in keys:
            if k.startswith("vqvae_LQ"):
                # LQ_count[k] = LQ_count[k] + 1
                # print("Deleting key {} from state_dict.".format(k))
                sd[k.replace('vqvae_LQ.', 'vqvae.')] = sd[k]
                del sd[k]
        
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict.".format(k))
        #             del sd[k]

        # import pdb
        # pdb.set_trace()

        state_dict = self.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        un_pretrained_keys = []
        count = 0
        for k in require_keys:
            if k not in keys:
                # miss 'vqvae.'
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                count = count + 1
                state_dict[k] = sd[k]

        print('State keys number =',len(state_dict))
        # count = 633
        print('keys loaded = ', count)
        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')
        self.load_state_dict(state_dict, strict=True)
        print(f"Restored from {path}")

    def init_from_ckpt_two(self, path_HQ, path_LQ, ignore_keys=list()):

        print('In two checkpoints load function-----------')

        # For associated HQ LQ weight
        HQ_ignore_keys = [ #'vqvae.encoder',
                        #   'vqvae.HQ_encoder',
                        #   'vqvae.quant_conv.weight',
                        #   'vqvae.quant_conv.bias',
                        #   'vqvae.HQ_quant_conv.weight',
                        #   'vqvae.HQ_quant_conv.bias',
                          'vqvae.decoder',
                          'loss.',
                        #   'vqvae.quantize.embedding.weight',
                          'vqvae.post_quant_conv.weight',
                          'vqvae.post_quant_conv.bias']
        
        LQ_ignore_keys = ['loss_LQ',
                          'vqvae_LQ.quantize.embedding.weight',
                          'vqvae_LQ.post_quant_conv.weight',
                          'vqvae_LQ.post_quant_conv.bias',
                          'vqvae_LQ.decoder']        
        HQ_replace_keys = ['vqvae.encoder',
                          'vqvae.quantize.embedding.weight',
                          'vqvae.quant_conv.weight',
                          'vqvae.quant_conv.bias']
        # LQ_ignore_keys = ['vqvae.decoder',
        #                   'loss',
        #                   'vqvae.quantize.embedding.weight',
        #                   'vqvae.post_quant_conv.weight',
        #                   'vqvae.post_quant_conv.bias']

        # From HQ checkpoint all: 460
        # From LQ checkpoint all: 921
        # 'vqvae.encoder': 170
        # 'vqvae.decoder': 230
        # 'loss': 55

        # under 128*128 size
        # From checkpoint all: 368
        # 'vqvae.encoder': 132
        # 'vqvae.decoder': 176
        # 'loss': 55
        HQ_count = {'vqvae.encoder': 0,
                    'vqvae.HQ_encoder': 0,
                    'vqvae.Fixed_encoder': 0,
                    'vqvae.decoder': 0,
                    'loss.': 0,
                    'vqvae.quant_conv.weight': 0,
                    'vqvae.quant_conv.bias': 0,
                    'vqvae.HQ_quant_conv.weight': 0,
                    'vqvae.HQ_quant_conv.bias': 0,
                    'vqvae.HQ_quantize.embedding.weight': 0,
                    'vqvae.Fixed_quant_conv.weight': 0,
                    'vqvae.Fixed_quant_conv.bias': 0,
                    'vqvae.Fixed_quantize.embedding.weight': 0,
                    'vqvae.quantize.embedding.weight': 0,
                    'vqvae.post_quant_conv.weight': 0,
                    'vqvae.post_quant_conv.bias': 0}
        # LQ_count = {'vqvae.encoder': 0,
        #             'vqvae.decoder': 0,
        #             'loss': 0,
        #             'vqvae.quant_conv.weight': 0,
        #             'vqvae.quant_conv.bias': 0,
        #             'vqvae.quantize.embedding.weight': 0,
        #             'vqvae.LQ_quantize.embedding.weight': 0,
        #             'vqvae.post_quant_conv.weight': 0,
        #             'vqvae.post_quant_conv.bias': 0}
        LQ_count = {'vqvae_LQ.encoder': 0,
                    'vqvae_LQ.decoder': 0,
                    'loss_LQ': 0,
                    'vqvae_LQ.quant_conv.weight': 0,
                    'vqvae_LQ.quant_conv.bias': 0,
                    'vqvae_LQ.quantize.embedding.weight': 0,
                    'vqvae_LQ.LQ_quantize.embedding.weight': 0,
                    'vqvae_LQ.post_quant_conv.weight': 0,
                    'vqvae_LQ.post_quant_conv.bias': 0}

        # load HQ checkpoint
        sd_HQ = torch.load(path_HQ, map_location="cpu")["state_dict"]
        HQ_keys = list(sd_HQ.keys())
        # load LQ checkpoint
        sd_LQ = torch.load(path_LQ, map_location="cpu")["state_dict"]
        LQ_keys = list(sd_LQ.keys())

        print('origin HQ keys number =',len(HQ_keys))
        print('origin LQ keys number =',len(LQ_keys))

        # import pdb
        # pdb.set_trace()

        # replace keys in HQ Keys
        for k in HQ_keys:
            for ik in HQ_replace_keys:
                if k.startswith(ik):
                    # HQ_count[ik] = HQ_count[ik] + 1
                    HQ_count[ik.replace('vqvae.', 'vqvae.HQ_')] = HQ_count[ik.replace('vqvae.', 'vqvae.HQ_')] + 1
                    HQ_count[ik.replace('vqvae.', 'vqvae.Fixed_')] = HQ_count[ik.replace('vqvae.', 'vqvae.Fixed_')] + 1
                    # print("Deleting key {} from state_dict.".format(k))
                    sd_LQ[k.replace('vqvae.', 'vqvae.HQ_')] = sd_LQ[k]
                    sd_HQ[k.replace('vqvae.', 'vqvae.Fixed_')] = sd_HQ[k]
                    del sd_LQ[k]
                    del sd_HQ[k]
        
        # delete keys in HQ Keys in LQ checkpoint
        for k in LQ_keys:
            for ik in HQ_ignore_keys:
                if k.startswith(ik):
                    HQ_count[ik] = HQ_count[ik] - 1
                    # HQ_count[ik.replace('vqvae.', 'vqvae.HQ_')] = HQ_count[ik.replace('vqvae.', 'vqvae.HQ_')] + 1
                    # print("Deleting key {} from state_dict.".format(k))
                    # sd_LQ[k.replace('vqvae.', 'vqvae.HQ_')] = sd_LQ[k]
                    del sd_LQ[k]

        # import pdb
        # pdb.set_trace()

        # delete and replace keys in LQ Keys (associated version)
        for k in LQ_keys:
            for ik in LQ_ignore_keys:
                if (k == 'vqvae_LQ.quantize.embedding.weight') and k.startswith(ik):
                    LQ_count['vqvae_LQ.quantize.embedding.weight'] = LQ_count['vqvae_LQ.quantize.embedding.weight'] - 1
                    LQ_count['vqvae_LQ.LQ_quantize.embedding.weight'] = LQ_count['vqvae_LQ.LQ_quantize.embedding.weight'] + 1
                    sd_LQ[k.replace('vqvae_LQ.', 'vqvae_LQ.LQ_')] = sd_LQ[k]
                    del sd_LQ[k]
                elif k.startswith(ik):
                    LQ_count[ik] = LQ_count[ik] - 1
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd_LQ[k]
        # stage one version--------------------------------------------------------
        # for k in LQ_keys:
        #     for ik in LQ_ignore_keys:
        #         if (k == 'vqvae.quantize.embedding.weight') and k.startswith(ik):
        #             LQ_count['vqvae.quantize.embedding.weight'] = LQ_count['vqvae.quantize.embedding.weight'] - 1
        #             LQ_count['vqvae.LQ_quantize.embedding.weight'] = LQ_count['vqvae.LQ_quantize.embedding.weight'] + 1
        #             sd_LQ[k.replace('vqvae.', 'vqvae.LQ_')] = sd_LQ[k]
        #             del sd_LQ[k]
        #         elif k.startswith(ik):
        #             LQ_count[ik] = LQ_count[ik] - 1
        #             # print("Deleting key {} from state_dict.".format(k))
        #             del sd_LQ[k]

        keys = list(sd_LQ.keys())
        for k in keys:
            if k.startswith("vqvae_LQ"):
                # LQ_count[k] = LQ_count[k] + 1
                # print("Deleting key {} from state_dict.".format(k))
                sd_LQ[k.replace('vqvae_LQ.', 'vqvae.')] = sd_LQ[k]
                del sd_LQ[k]
        
        # HQ keys = 460
        # LQ keys = 346+1
        # import pdb
        # pdb.set_trace()

        state_dict = self.state_dict()
        require_keys = state_dict.keys()
        HQ_keys = sd_HQ.keys()
        LQ_keys = sd_LQ.keys()
        print('processed HQ keys number =',len(HQ_keys))
        print('processed LQ keys number =',len(LQ_keys))
        un_pretrained_keys = []
        count = 0
        HQ_Key_count = 0
        LQ_Key_count = 0
        for k in require_keys:
            # HQ checkpoint
            if k not in HQ_keys:
                # miss 'vqvae.'
                if k[6:] in HQ_keys:
                    state_dict[k] = sd_HQ[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd_HQ[k]
                HQ_Key_count = HQ_Key_count + 1
                count = count + 1
            # LQ checkpoint
            if k not in LQ_keys:
                # miss 'vqvae.'
                if k[6:] in LQ_keys:
                    state_dict[k] = sd_LQ[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd_LQ[k]
                LQ_Key_count = LQ_Key_count + 1
                count = count + 1
 
        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')

        self.load_state_dict(state_dict, strict=True)
        # count = 633+1
        print(f"Number of HQ weight loaded = {HQ_Key_count}") # 405
        print(f"Number of LQ weight loaded = {LQ_Key_count}") # 346
        print(f"Number of all weight loaded = {count}")       # 751  
        print(f"Restored from {path_HQ} and {path_LQ}")

    def init_cond_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.cond_stage_model = model
    
    def forward(self, input, gt=None):
        # dec, diff, info, hs = self.vqvae(input)
        # return dec, diff, info, hs
        
        # import pdb
        # pdb.set_trace()

        # quant_z, z_indices, z_info, z_hs, z_h, z_dictionary = self.encode_to_z(input, self.encoder_codebook_type)
        if gt is not None:
            quant_gt, gt_indices, gt_info, gt_hs, gt_h, gt_dictionary = self.encode_to_gt(gt)

        # codeformer code --------------------------------

        # LQ feature from LQ encoder
        z_hs = self.vqvae.encoder(input)
        z_h = self.vqvae.quant_conv(z_hs['out'])
        # LQ feature from HQ encoder
        z_hs_HQ = self.vqvae.HQ_encoder(input)
        z_h_HQ = self.vqvae.HQ_quant_conv(z_hs_HQ['out'])
        
        # concate features
        # concat_feature = torch.cat((z_h, z_h_HQ), 1)
        # Linear Projection
        # out_feature = self.linear_projection(concat_feature)
        
        # Cross attention
        out_feature = self.cross_attention(z_h_HQ, z_h)
        
        # origin HQ codebook for index
        quant_z, emb_loss, z_info, z_dictionary = self.vqvae.Fixed_quantize(z_h)
        indices = z_info[2].view(quant_z.shape[0], -1)
        z_indices = self.permuter(indices)

        if gt is None:
            quant_gt = quant_z
            gt_indices = z_indices

        lq_feat = out_feature
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,z_h.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # import pdb
        # pdb.set_trace()

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        BCE_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gt_indices.view(-1))

        code_only=False
        if code_only: # for training stage II
          # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        # get index from origin HQ codebook
        quant_feat = self.vqvae.Fixed_quantize.get_codebook_entry(top_idx.reshape(-1), shape=[z_h.shape[0],16,16,256])
        
        L2_loss = F.mse_loss(lq_feat, quant_gt)
        
        # preserve gradients
        quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        dec = self.vqvae.decode(quant_feat)

        # codeformer code --------------------------------

        # cz_indices = torch.cat((z_indices, z_indices), dim=1)
        # logits, loss = self.transformer(z_indices, targets=gt_indices)
        # logits = logits[:, gt_indices.shape[1]-1:]
        # temperature = 1.0
        # logits = logits / temperature
        # probs = F.softmax(logits, dim=-1)
        # _, index = torch.topk(probs, k=1, dim=-1)
        # bhwc = (quant_z.shape[0],quant_z.shape[2],quant_z.shape[3],quant_z.shape[1])
        # quant_feat = self.vqvae.quantize.get_codebook_entry(
        #     index.reshape(-1), shape=bhwc)
        # quant_feat = quant_z + (quant_feat - quant_z).detach()
        # dec = self.vqvae.decode(quant_feat)

        # dec, diff, info, hs, h, quant, dictionary = self.vqvae(input)
        # return dec, diff, info, hs, h, quant, dictionary
        return dec, BCE_loss, L2_loss, z_info, z_hs, z_h, quant_z, z_dictionary

    @torch.no_grad()
    def encode_to_z(self, x, encoder_codebook_type):
        # import pdb
        # pdb.set_trace()
        if (encoder_codebook_type is not None) and (encoder_codebook_type == 'LQHQ'):
            quant_z, _, info, hs, h, dictionary = self.vqvae.encode(x)
        if (encoder_codebook_type is not None) and (encoder_codebook_type == 'LQLQ'):
            quant_z, _, info, hs, h, dictionary = self.vqvae.LQ_encode(x)    
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices, info, hs, h, dictionary

    @torch.no_grad()
    def encode_to_gt(self, gt):
        # import pdb
        # pdb.set_trace()
        quant_gt, _, info, hs, h, dictionary = self.vqvae.HQ_encode(gt)
        indices = info[2].view(quant_gt.shape[0], -1)
        indices = self.permuter(indices)
        return quant_gt, indices, info, hs, h, dictionary

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        if optimizer_idx == None:
            optimizer_idx = 0

        x = batch[self.image_key]
        gt = batch['gt']
        xrec, BCE_loss, L2_loss, info, hs,_,_,_ = self(x, gt)

        qloss = BCE_loss + 10*L2_loss

        if self.image_key != 'gt':
            x = batch['gt']

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(
                x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        if optimizer_idx == 0:
            
            aeloss = BCE_loss + 10*L2_loss
            
            rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous()))

            log_dict_ae = {
                   "train/BCE_loss": BCE_loss.detach().mean(),
                   "train/L2_loss": L2_loss.detach().mean(),
                   "train/Rec_loss": rec_loss.detach().mean()
                }
            
            # autoencode
            # aeloss, log_dict_ae = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
            #                                 last_layer=self.get_last_layer(), split="train")

            bce_loss = log_dict_ae["train/BCE_loss"]
            self.log("BCE_loss", bce_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            
            l2_loss = log_dict_ae["train/L2_loss"]
            self.log("L2_loss", l2_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            
            Rec_loss = log_dict_ae["train/Rec_loss"]
            self.log("Rec_loss", Rec_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)

            # self.log("train/aeloss", aeloss, prog_bar=True,
            #          logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return discloss

        if self.disc_start <= self.global_step:

            # left eye
            if optimizer_idx == 2:
                # discriminator
                disc_left_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                          last_layer=None, split="train")
                self.log("train/disc_left_loss", disc_left_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_left_loss

            # right eye
            if optimizer_idx == 3:
                # discriminator
                disc_right_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_right_loss", disc_right_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_right_loss

            # mouth
            if optimizer_idx == 4:
                # discriminator
                disc_mouth_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_mouth_loss", disc_mouth_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_mouth_loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.image_key]
        gt = batch['gt']
        xrec, BCE_loss, L2_loss, info, hs,_,_,_ = self(x, gt)

        qloss = BCE_loss + L2_loss

        if self.image_key != 'gt':
            x = batch['gt']

        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, None, 0, self.global_step,
        #                                 last_layer=self.get_last_layer(), split="val")

        # discloss, log_dict_disc = self.loss(qloss, x, xrec, None, 1, self.global_step,
        #                                     last_layer=None, split="val")
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)

        rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous()))

        log_dict_ae = {
                "val/BCE_loss": BCE_loss.detach().mean(),
                "val/L2_loss": L2_loss.detach().mean(),
                "val/Rec_loss": rec_loss.detach().mean()
            }
        self.log_dict(log_dict_ae)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        # under 128*128 size
        # From checkpoint all: 368
        # 'vqvae.encoder': 132
        # 'vqvae.decoder': 176
        # 'loss': 55

        normal_params = []
        special_params = []
        fixed_params = []
        fixed_parameter = 0
        test_count = 0
        # autoencoder part -------------------------------
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue
            # if 'decoder' in name and 'attn' in name:
            #     special_params.append(param)
            if 'Fixed' in name:
                special_params.append(param)
                fixed_parameter = fixed_parameter + 1
                continue
            if 'decoder' in name or 'post_quant_conv' in name or 'quantize' in name:
                test_count = test_count + 1
                # continue
                special_params.append(param)
                # print(name)
            else:
                normal_params.append(param)
        # import pdb
        # pdb.set_trace()
        
        # self.linear_projection
        # for name, param in self.linear_projection.named_parameters():
        #     if not param.requires_grad:
        #         continue
        #     else:
        #         normal_params.append(param) 

        # self.cross_attention
        for name, param in self.cross_attention.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 

        # transformer part--------------------------------
        
        # self.position_emb
        # self.feat_emb
        # self.ft_layers
        # self.idx_pred_layer
        
        normal_params.append(self.position_emb)   
        
        for name, param in self.feat_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 

        for name, param in self.ft_layers.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 

        for name, param in self.idx_pred_layer.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)                 
        
        # import pdb
        # pdb.set_trace()
        
        # print('special_params', special_params)
        opt_ae_params = [{'params': normal_params, 'lr': lr}]
                        #  {'params': special_params, 'lr': lr*self.special_params_lr_scale}]
                        # {'params': special_params, 'lr': 0.0}]
        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9))

        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))

        # optimizations = [opt_ae, opt_disc]
        optimizations = opt_ae

        # s0 = torch.optim.lr_scheduler.MultiStepLR(
        #     opt_ae, milestones=self.schedule_step, gamma=0.1, verbose=True)
        # s1 = torch.optim.lr_scheduler.MultiStepLR(
        #     opt_disc, milestones=self.schedule_step, gamma=0.1, verbose=True)
        # schedules = [s0, s1]
        # schedules = s0

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]

            s2 = torch.optim.lr_scheduler.MultiStepLR(
                opt_l, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s3 = torch.optim.lr_scheduler.MultiStepLR(
                opt_r, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s4 = torch.optim.lr_scheduler.MultiStepLR(
                opt_m, milestones=self.schedule_step, gamma=0.1, verbose=True)
            schedules += [s2, s3, s4]

        # return optimizations, schedules
        return optimizations

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv.weight
        return self.vqvae.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch[self.image_key]
        x = x.to(self.device)
        gt = batch['gt'].to(self.device)
        xrec, _, _, _, _, _, _, _ = self(x, gt)
        log["inputs"] = x
        log["reconstructions"] = xrec

        if self.image_key != 'gt':
            x = batch['gt']
            log["gt"] = x
        return log
