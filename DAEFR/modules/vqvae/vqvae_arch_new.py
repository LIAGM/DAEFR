import torch
import torch.nn as nn
import random
import math
import torch.nn.functional as F
import numpy as np
# from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.utils.spectral_norm as SpectralNorm
import DAEFR.distributed as dist_fn

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # import pdb
        # pdb.set_trace()
        
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        # z_flattened -> ( batch*height*width, e_dim = 256)
        z_flattened = z.view(-1, self.e_dim)
        
        # distances d from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        # d shape -> ( batch*height*width, n_e = 1024)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        
        # min_value shape -> (batch*height*width)
        # min_encoding_indices -> (batch*height*width) 
        # "min_encoding_indices" indicate the corresponding code items
        min_value, min_encoding_indices = torch.min(d, dim=1)
        
        # min_encoding_indices -> (batch*height*width, 1)
        min_encoding_indices = min_encoding_indices.unsqueeze(1)

        # min_encodings shape -> ( batch*height*width, n_e = 1024)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        # torch.matmul(min_encodings, self.embedding.weight) 
        # shape -> ( batch*height*width, e_dim = 256)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d), self.embedding.weight

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

# pytorch_diffusion + derived encoder decoder
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        
        # import pdb
        # pdb.set_trace()

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.num = 0

    def forward(self, x, y=None):
        
        # import pdb
        # pdb.set_trace()
        
        h_ = x
        h_ = self.norm1(h_)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b, self.head_size, self.att_size ,h*w) 
        q = q.permute(0, 3, 1, 2) # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size ,h*w) 
        k = k.permute(0, 3, 1, 2)

        v = v.reshape(b, self.head_size, self.att_size ,h*w) 
        v = v.permute(0, 3, 1, 2)


        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2,3)

        scale = int(self.att_size)**(-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous() # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2)

        w_ = self.proj_out(w_)

        return x+w_


# CFT control module ----------------------------------------------------------
class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResnetBlock(in_channels=2*in_ch,
                                      out_channels=out_ch,
                                      temb_channels=0,
                                      dropout=0.0)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1), None)
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class MultiHeadEncoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, double_z=True, enable_mid=True,
                 head_size=1, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        # for skip connection ----------------------------------------
        self.connect_list = ['32', '64', '128', '256']

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        self.fuse_encoder_block = {'256':1, '128':2, '64':3, '32':4}

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, feature_scale=0):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # import pdb
        # pdb.set_trace()

        hs = {}
        # timestep embedding
        temb = None

        # for skip connection --------------------------------
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]

        # import pdb
        # pdb.set_trace()

        # downsampling
        h = self.conv_in(x)
        hs['in'] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            # for skip connection -------------------------
            if i_level in out_list:
                enc_feat_dict[str(h.shape[-1])] = h.clone()
            
            if i_level != self.num_resolutions-1:
                # hs.append(h)
                hs['block_'+str(i_level)] = h
                h = self.down[i_level].downsample(h)

        # import pdb
        # pdb.set_trace()
        
        # middle
        # h = hs[-1]
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs['block_'+str(i_level)+'_atten'] = h
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            hs['mid_atten'] = h

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # hs.append(h)
        hs['out'] = h

        if feature_scale > 0:
            return hs, enc_feat_dict

        return hs

class MultiHeadDecoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=16, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # for skip connection ---------------------------
        self.connect_list = ['32', '64', '128', '256']

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        
        # CFT control module
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)


        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, enc_feat_dict=None, feature_scale=0):
        
        # import pdb
        # pdb.set_trace()
        
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # get skip connection features
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            
            # import pdb
            # pdb.set_trace()

            # for skip connection
            if i_level in fuse_list and enc_feat_dict is not None: # fuse after i-th block
                f_size = str(h.shape[-1])
                if feature_scale > 0:
                    h = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), h, feature_scale)
            
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class MultiHeadDecoderTransformer(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=16, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs):
        #assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # import pdb
        # pdb.set_trace()

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h, hs['mid_atten'])
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, hs['block_'+str(i_level)+'_atten'])
                    # hfeature = h.clone()
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQVAEGAN(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(VQVAEGAN, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary

class VQVAEGANMultiHeadTransformer(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, constrastive_learning_loss_weight=0.0,
                 head_size=1):
        super(VQVAEGANMultiHeadTransformer, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        self.decoder = MultiHeadDecoderTransformer(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def encode(self, x):
        
        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant, hs):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, hs)

        return dec

    def forward(self, input):
        quant, diff, info, hs = self.encode(input)
        dec = self.decode(quant, hs)

        return dec, diff, info, hs    


class VQVAEGANTWO(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(VQVAEGANTWO, self).__init__()

        #LQ
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        self.HQ_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)                       
        #HQ
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        # Origin_HQ_codebook
        self.HQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Associated HQ_codebook
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Associated LQ_codebook
        self.LQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        #LQ
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        #HQ
        self.HQ_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        #HQ
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_encoder.named_parameters():
                param.requires_grad = False    
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_quant_conv.named_parameters():
                param.requires_grad = False 

    # HQ encoder + origin HQ codebook
    def HQ_encode(self, x):

        hs = self.HQ_encoder(x)
        h = self.HQ_quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.HQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary
    
    # LQ encoder + Associated LQ codebook
    def LQ_encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.LQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    # LQ encoder + Associated HQ codebook
    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary         



# For HQ LQ feature association
class VQVAEGANLQ(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(VQVAEGANLQ, self).__init__()

        # LQ encoder
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        # HQ encoder
        self.HQ_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)

        # LQ decoder
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        # LQ codebook
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        # LQ
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        # HQ
        self.HQ_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        # LQ
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary




class VQVAEGANMERGE(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(VQVAEGANMERGE, self).__init__()

        #LQ
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        self.HQ_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size) 
        # For ground truth
        self.Fixed_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)                       
        #HQ
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        # Origin_HQ_codebook
        self.Fixed_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Associated HQ_codebook
        self.HQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Associated LQ_codebook
        self.LQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        #LQ
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        #HQ
        self.HQ_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        # Fixed for ground truth
        self.Fixed_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        #HQ
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.Fixed_quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.Fixed_quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_encoder.named_parameters():
                param.requires_grad = False    
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_quant_conv.named_parameters():
                param.requires_grad = False 

    # HQ encoder + origin HQ codebook
    def HQ_encode(self, x):

        hs = self.Fixed_encoder(x)
        h = self.Fixed_quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.Fixed_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary
    
    # LQ encoder + Associated LQ codebook
    def LQ_encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.LQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    # LQ encoder + Associated HQ codebook
    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.HQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary     


class VQVAEGANCODE(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(VQVAEGANCODE, self).__init__()

        
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        # For ground truth
        self.Fixed_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)                       
        
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        # Origin_HQ_codebook
        self.Fixed_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Origin_HQ_codebook
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)

        # Fixed for ground truth
        self.Fixed_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.Fixed_quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.Fixed_quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_encoder.named_parameters():
                param.requires_grad = False    
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_quant_conv.named_parameters():
                param.requires_grad = False 

    # HQ encoder + origin HQ codebook
    def HQ_encode(self, x):

        hs = self.Fixed_encoder(x)
        h = self.Fixed_quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.Fixed_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary


    def encode(self, x):
        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary 



# With finetune control module ------------------------------------------------------
class VQVAEGANMERGEWITHCFT(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, **ignore_kwargs):
        super(VQVAEGANMERGEWITHCFT, self).__init__()

        #LQ
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        self.HQ_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size) 
        # For ground truth
        self.Fixed_encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)                       
        #HQ
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        # Origin_HQ_codebook
        self.Fixed_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Associated HQ_codebook
        self.HQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Associated LQ_codebook
        self.LQ_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        #LQ
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        #HQ
        self.HQ_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        # Fixed for ground truth
        self.Fixed_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        #HQ
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.Fixed_quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.Fixed_quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_encoder.named_parameters():
                param.requires_grad = False    
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ_quant_conv.named_parameters():
                param.requires_grad = False 

    # HQ encoder + origin HQ codebook
    def HQ_encode(self, x):

        hs = self.Fixed_encoder(x)
        h = self.Fixed_quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.Fixed_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary
    
    # LQ encoder + Associated LQ codebook
    def LQ_encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.LQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    # LQ encoder + Associated HQ codebook
    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info, dictionary = self.HQ_quantize(h)
        # return quant, emb_loss, info, hs
        return quant, emb_loss, info, hs, h, dictionary

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        # quant, diff, info, hs = self.encode(input)
        quant, diff, info, hs, h, dictionary = self.encode(input)
        dec = self.decode(quant)

        # return dec, diff, info, hs
        # info = (perplexity, min_encodings, min_encoding_indices, d)
        return dec, diff, info, hs, h, quant, dictionary

