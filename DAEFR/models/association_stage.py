import os, math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from main_DAEFR import instantiate_from_config
import numpy as np

from DAEFR.modules.vqvae.utils import get_roi_regions



class DAEFRModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path_HQ=None,
                 ckpt_path_LQ=None,
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
        self.vqvae_LQ = instantiate_from_config(ddconfig)


        lossconfig['params']['distill_param'] = ddconfig['params']
        self.loss = instantiate_from_config(lossconfig)
        self.loss_LQ = instantiate_from_config(lossconfig)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # get the weights from HQ and LQ checkpoints
        if (ckpt_path_HQ is not None) and (ckpt_path_LQ is not None):
            self.init_from_ckpt_two(
                ckpt_path_HQ, ckpt_path_LQ, ignore_keys=ignore_keys)

        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']

        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step



    def init_from_ckpt_two(self, path_HQ, path_LQ, ignore_keys=list()):

        print('In two checkpoints load function-----------')

        HQ_ignore_keys = []
        LQ_ignore_keys = []

        # From checkpoint all: 460
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
        LQ_count = {'vqvae.encoder': 0,
                    'vqvae.decoder': 0,
                    'loss': 0,
                    'vqvae.quant_conv.weight': 0,
                    'vqvae.quant_conv.bias': 0,
                    'vqvae.quantize.embedding.weight': 0,
                    'vqvae.LQ_quantize.embedding.weight': 0,
                    'vqvae.post_quant_conv.weight': 0,
                    'vqvae.post_quant_conv.bias': 0}

        # load HQ checkpoint
        sd_HQ = torch.load(path_HQ, map_location="cpu")["state_dict"]
        HQ_keys = list(sd_HQ.keys())
        # load LQ checkpoint
        sd_LQ = torch.load(path_LQ, map_location="cpu")["state_dict"]
        LQ_keys = list(sd_LQ.keys())

        print('HQ keys number =',len(HQ_keys))
        print('LQ keys number =',len(LQ_keys))


        # replace the keys in LQ checkpoint
        for k in LQ_keys:
            if k.startswith('vqvae'):
            # print("Deleting key {} from state_dict.".format(k))
                sd_LQ[k.replace('vqvae', "vqvae_LQ")] = sd_LQ[k]
                del sd_LQ[k]
            elif k.startswith('loss'):
                sd_LQ['loss_LQ'+k[4:]] = sd_LQ[k]
                del sd_LQ[k]

        # HQ keys = 460
        # LQ keys = 460

        state_dict = self.state_dict()
        require_keys = state_dict.keys()
        HQ_keys = sd_HQ.keys()
        LQ_keys = sd_LQ.keys()
        un_pretrained_keys = []
        HQ_count = 0
        LQ_count = 0
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
                HQ_count = HQ_count + 1
            # LQ checkpoint
            if k not in LQ_keys:
                # miss 'vqvae.'
                if k[6:] in LQ_keys:
                    state_dict[k] = sd_LQ[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd_LQ[k]
                LQ_count = LQ_count + 1
 
        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')

        self.load_state_dict(state_dict, strict=True)
        # HQ_count = 460
        # LQ_count = 460
        print(f"Number of HQ weight loaded = {HQ_count}")
        print(f"Number of LQ weight loaded = {LQ_count}")
        print(f"Restored from {path_HQ} and {path_LQ}")
    
    def forward(self, input, gt=None):
        # dec, diff, info, hs = self.vqvae(input)
        # return dec, diff, info, hs

        # HQ part 
        HQ_hs = self.vqvae.encoder(gt)
        HQ_h = self.vqvae.quant_conv(HQ_hs['out'])
        HQ_quant, HQ_emb_loss, HQ_info, HQ_dictionary = self.vqvae.quantize(HQ_h)

        # LQ part 
        LQ_hs = self.vqvae_LQ.encoder(input)
        LQ_h = self.vqvae_LQ.quant_conv(LQ_hs['out'])
        LQ_quant, LQ_emb_loss, LQ_info, LQ_dictionary = self.vqvae_LQ.quantize(LQ_h)

        # Do CLIP like loss calculation
        # reshape z -> (batch, height, width, channel) and flatten
        HQ_z = HQ_h.permute(0, 2, 3, 1).contiguous()
        # z_flattened -> ( batch*height*width, e_dim = 256)
        HQ_z_flattened = HQ_z.view(-1, HQ_z.shape[3])
        # reshape z -> (batch, height, width, channel) and flatten
        LQ_z = LQ_h.permute(0, 2, 3, 1).contiguous()
        # z_flattened -> ( batch*height*width, e_dim = 256)
        LQ_z_flattened = LQ_z.view(-1, LQ_z.shape[3])

        # normalized features
        HQ_z_flattened = HQ_z_flattened / HQ_z_flattened.norm(dim=1, keepdim=True)
        LQ_z_flattened = LQ_z_flattened / LQ_z_flattened.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_HQ_image = logit_scale * HQ_z_flattened @ LQ_z_flattened.t()
        logits_LQ_image = logits_HQ_image.t()
        
        ground_truth = torch.arange(logits_HQ_image.shape[0], dtype=torch.long, device=self.device)

        loss_HQ_img = nn.CrossEntropyLoss()
        loss_LQ_img = nn.CrossEntropyLoss()
        HQ_loss = loss_HQ_img(logits_HQ_image, ground_truth)
        LQ_loss = loss_LQ_img(logits_LQ_image, ground_truth)
        CLIP_loss = (HQ_loss+LQ_loss) / 2

        HQ_dec = self.vqvae.decode(HQ_quant)
        LQ_dec = self.vqvae_LQ.decode(LQ_quant)

        return HQ_dec, LQ_dec, CLIP_loss, HQ_emb_loss, LQ_emb_loss, HQ_quant, LQ_quant

    @torch.no_grad()
    def encode_to_gt(self, gt):
        quant_gt, _, info, hs, h, dictionary = self.vqvae.HQ_encode(gt)
        indices = info[2].view(quant_gt.shape[0], -1)
        return quant_gt, indices, info, hs, h, dictionary

    def training_step(self, batch, batch_idx, optimizer_idx):

        # LQ image
        x = batch[self.image_key]
        # HQ image
        gt = batch['gt']
        # HQ_dec, LQ_dec, CLIP_loss, HQ_emb_loss, LQ_emb_loss, HQ_quant, LQ_quant
        # xrec, qloss, info, hs,_,_,_ = self(x, gt)
        HQ_xrec, LQ_xrec, CLIP_loss, HQ_qloss, LQ_qloss,_,_ = self(x, gt)

        # if self.image_key != 'gt':
        #     x = batch['gt']

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = HQ_xrec.shape[-1] / 512
            components = get_roi_regions(
                x, HQ_xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        # HQ part
        if optimizer_idx == 0:
            # HQ autoencode
            HQ_aeloss, HQ_log_dict_ae = self.loss(HQ_qloss, gt, HQ_xrec, components, optimizer_idx, self.global_step,
                                            CLIP_loss,
                                            last_layer=self.get_last_layer(), split="train")

            HQ_CLIP_loss = HQ_log_dict_ae["train/CLIP_loss"]
            self.log("Associate_loss", HQ_CLIP_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)

            HQ_Rec_loss = HQ_log_dict_ae["train/rec_loss"]
            self.log("HQ_Rec_loss", HQ_Rec_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)

            self.log("train/HQ_aeloss", HQ_aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(HQ_log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return HQ_aeloss

        if optimizer_idx == 1:
            # HQ discriminator
            HQ_discloss, HQ_log_dict_disc = self.loss(HQ_qloss, gt, HQ_xrec, components, optimizer_idx, self.global_step,
                                                CLIP_loss,
                                                last_layer=None, split="train")
            self.log("train/HQ_discloss", HQ_discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(HQ_log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return HQ_discloss

        # LQ part
        if optimizer_idx == 2:
            # LQ autoencode
            LQ_aeloss, LQ_log_dict_ae = self.loss_LQ(LQ_qloss, x, LQ_xrec, components, optimizer_idx, self.global_step,
                                            CLIP_loss,
                                            last_layer=self.get_last_layer_LQ(), split="train")

            LQ_CLIP_loss = LQ_log_dict_ae["train/CLIP_loss"]
            self.log("LQ_Associate_loss", LQ_CLIP_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            
            LQ_Rec_loss = LQ_log_dict_ae["train/rec_loss"]
            self.log("LQ_Rec_loss", LQ_Rec_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            
            self.log("train/LQ_aeloss", LQ_aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(LQ_log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return LQ_aeloss

        if optimizer_idx == 3:
            # LQ discriminator
            LQ_discloss, LQ_log_dict_disc = self.loss_LQ(LQ_qloss, x, LQ_xrec, components, optimizer_idx, self.global_step,
                                                CLIP_loss,
                                                last_layer=None, split="train")
            self.log("train/LQ_discloss", LQ_discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(LQ_log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return LQ_discloss

    def validation_step(self, batch, batch_idx):
        x = batch[self.image_key]
        gt = batch['gt']

        HQ_xrec, LQ_xrec, CLIP_loss, HQ_qloss, LQ_qloss,_,_ = self(x, gt)

        # if self.image_key != 'gt':
        #     x = batch['gt']

        # HQ part
        HQ_aeloss, HQ_log_dict_ae = self.loss(HQ_qloss, gt, HQ_xrec, None, 0, self.global_step,
                                        CLIP_loss,
                                        last_layer=self.get_last_layer(), split="val")

        HQ_discloss, HQ_log_dict_disc = self.loss(HQ_qloss, gt, HQ_xrec, None, 1, self.global_step,
                                            CLIP_loss,
                                            last_layer=None, split="val")

        # LQ part
        LQ_aeloss, LQ_log_dict_ae = self.loss_LQ(LQ_qloss, x, LQ_xrec, None, 2, self.global_step,
                                        CLIP_loss,
                                        last_layer=self.get_last_layer_LQ(), split="val")

        LQ_discloss, LQ_log_dict_disc = self.loss_LQ(LQ_qloss, x, LQ_xrec, None, 3, self.global_step,
                                            CLIP_loss,
                                            last_layer=None, split="val")         

        HQ_rec_loss = HQ_log_dict_ae["val/rec_loss"]
        self.log("val/HQ_rec_loss", HQ_rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/HQ_aeloss", HQ_aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        LQ_rec_loss = LQ_log_dict_ae["val/rec_loss"]
        self.log("val/LQ_rec_loss", LQ_rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/LQ_aeloss", LQ_aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        self.log_dict(HQ_log_dict_ae)
        self.log_dict(HQ_log_dict_disc)
        self.log_dict(LQ_log_dict_ae)
        self.log_dict(LQ_log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        # HQ part ---------------------------------------------------------------
        HQ_normal_params = []
        HQ_special_params = []
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue
            # if 'decoder' in name and 'attn' in name:
            #     special_params.append(param)
            if 'decoder' in name or 'post_quant_conv' in name:
                HQ_special_params.append(param)
                # print(name)
            else:
                HQ_normal_params.append(param)
        
        # print('special_params', special_params)
        opt_ae_params_HQ = [{'params': HQ_normal_params, 'lr': lr},
                         {'params': HQ_special_params, 'lr': lr*self.special_params_lr_scale}]
                        # {'params': special_params, 'lr': 0.0}]
        opt_ae_HQ = torch.optim.Adam(opt_ae_params_HQ, betas=(0.5, 0.9))

        opt_disc_HQ = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        # LQ part ---------------------------------------------------------------
        LQ_normal_params = []
        LQ_special_params = []
        for LQ_name, LQ_param in self.vqvae_LQ.named_parameters():
            if not param.requires_grad:
                continue
            # if 'decoder' in name and 'attn' in name:
            #     special_params.append(param)
            if 'decoder' in LQ_name or 'post_quant_conv' in LQ_name:
                LQ_special_params.append(LQ_param)
                # print(name)
            else:
                LQ_normal_params.append(LQ_param)
        
        # print('special_params', special_params)
        opt_ae_params_LQ = [{'params': LQ_normal_params, 'lr': lr},
                         {'params': LQ_special_params, 'lr': lr*self.special_params_lr_scale}]
                        # {'params': special_params, 'lr': 0.0}]
        opt_ae_LQ = torch.optim.Adam(opt_ae_params_LQ, betas=(0.5, 0.9))

        opt_disc_LQ = torch.optim.Adam(self.loss_LQ.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        optimizations = [opt_ae_HQ, opt_disc_HQ, opt_ae_LQ, opt_disc_LQ]

        # HQ part ----------
        s0 = torch.optim.lr_scheduler.MultiStepLR(
            opt_ae_HQ, milestones=self.schedule_step, gamma=0.1, verbose=True)
        s1 = torch.optim.lr_scheduler.MultiStepLR(
            opt_disc_HQ, milestones=self.schedule_step, gamma=0.1, verbose=True)
        # LQ part ----------
        s2 = torch.optim.lr_scheduler.MultiStepLR(
            opt_ae_LQ, milestones=self.schedule_step, gamma=0.1, verbose=True)
        s3 = torch.optim.lr_scheduler.MultiStepLR(
            opt_disc_LQ, milestones=self.schedule_step, gamma=0.1, verbose=True)    
        
        schedules = [s0, s1, s2, s3]



        return optimizations, schedules

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv.weight
        return self.vqvae.decoder.conv_out.weight

    def get_last_layer_LQ(self):
        if self.fix_decoder:
            return self.vqvae_LQ.quant_conv.weight
        return self.vqvae_LQ.decoder.conv_out.weight    

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch[self.image_key]
        x = x.to(self.device)
        gt = batch['gt'].to(self.device)
        # HQ_dec, LQ_dec, CLIP_loss, HQ_emb_loss, LQ_emb_loss, HQ_quant, LQ_quant
        # xrec, _, _, _, _, _, _ = self(x, gt)
        HQ_dec, LQ_dec, _, _, _, _, _ = self(x, gt)
        
        log["HQ_inputs"] = gt
        log["LQ_inputs"] = x
        log["HQ_reconstructions"] = HQ_dec
        log["LQ_reconstructions"] = LQ_dec

        # if self.image_key != 'gt':
        #     x = batch['gt']
        #     log["gt"] = x
        return log
