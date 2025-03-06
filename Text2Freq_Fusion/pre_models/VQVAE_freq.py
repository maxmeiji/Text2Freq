import torch
import torch.nn as nn
import torch.nn.functional as F
from pre_utils.encdec_freq import Encoder, Decoder
from pre_utils.vqquantizer import VectorQuantizer

class VQVAE_freq(nn.Module):
    def __init__(self, hidden_dim, series_length, n_embed, embed_dim, stride):
        super(VQVAE_freq, self).__init__()
        self.series_length = series_length
        # in_channel: the channel num is 2 when the input is frequency-domain
        self.encoder = Encoder(in_channel = 2, hidden_dim=hidden_dim, n_res_blocks=2, input_length = series_length//2, stride = stride)
        self.decoder = Decoder(out_channel = 2, hidden_dim=hidden_dim, n_res_blocks=2, input_length = series_length//2, stride = stride)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.5)
        self.post_quant_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def encode(self, x):
        # print(f'shape of input series {x.shape}') # [b, c, f] = [16, 2, 6]
        h = self.encoder(x) 
        # print(f'shape of output series {h.shape}') # [b, c_d, f] = [16, 16, 6]
        quant, emb_loss, _, _ = self.quantize(h)  
        # print(f'shape of discrete quant {quant.shape}') # [b, e_d f] = [16, 16, 6]

        return quant, emb_loss

    def decode(self, quant):
        # quant shape [batch, hidden size, series length]
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant) # [b, 2, 6]
        
        return dec

    def forward(self, input):
        quant, diff = self.encode(input)
        dec = self.decode(quant)
        # print(f'decoder shape {dec.shape}')
        return dec, diff

    def loss_function(self, reconstructed_series, series, diff):
        recon_loss = F.mse_loss(reconstructed_series, series, reduction='mean')
        return recon_loss + diff, recon_loss
    
    def get_codebook_entry(self, indices):
        return self.quantize.get_codebook_entry(indices)
