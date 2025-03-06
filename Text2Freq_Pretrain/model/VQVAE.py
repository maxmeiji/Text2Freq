import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encdec import Encoder, Decoder
from utils.vqquantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, hidden_dim, series_length, n_embed, embed_dim, stride):
        super(VQVAE, self).__init__()
        self.series_length = series_length
        self.encoder = Encoder(hidden_dim=hidden_dim, n_res_blocks=2, input_length = series_length, stride = stride)
        self.decoder = Decoder(hidden_dim=hidden_dim, n_res_blocks=2, input_length = series_length, stride = stride)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.5)
        self.post_quant_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def encode(self, x):
        # print(f'shape of input series {x.shape}') [b, s] = [16,12]
        h = self.encoder(x) 
        # print(f'shape of output series {h.shape}') [b,s] = [16, 32, 12]
        quant, emb_loss, _, _ = self.quantize(h)  
        # print(f'shape of discrete quant {quant.shape}') [b, s] = [16, 32, 12]

        return quant, emb_loss

    def decode(self, quant):
        # quant shape [batch, hidden size, series length]
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        
        return dec

    def forward(self, input):
        quant, diff = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def loss_function(self, reconstructed_series, series, diff):
        recon_loss = F.mse_loss(reconstructed_series, series, reduction='mean')
        return recon_loss + diff, recon_loss
    
    def get_codebook_entry(self, indices):
        return self.quantize.get_codebook_entry(indices)
