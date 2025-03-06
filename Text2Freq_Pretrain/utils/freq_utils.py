import torch

def time_to_freq(x):
    # series [batch size, series_length]
    B, L = x.shape
    x = x.view(B,1,L)
    x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
    
    # view as real number -> [b, 1, d, 2]
    x = torch.view_as_real(x) 

    # since the first frequency is DC, and the input value has been normalized, here we can discard the first freq
    x = x[:,:,1:,:]

    # rearrange channel dimension -> [b, 2, d]
    x  = x.permute(0,3,2,1).squeeze(-1)

    return x

def freq_to_time(x):
    # freq [batch size, 2, d]
    B, C, D= x.shape

    # add dc_component back 
    dc_component = torch.zeros(B, 2, 1, device=x.device, dtype=x.dtype)
    x = torch.cat([dc_component, x], dim=2)
    
    x = x.permute(0,2,1).contiguous()
    x = torch.view_as_complex(x)
    
    x = torch.fft.irfft(x, dim=1, norm='ortho') # iFFT on N dimension

    return x


def padding_for_lf(x, lf=6):
    # x.shape b,2,d
    x_lf = torch.zeros_like(x)
    x_lf[:,:,:lf] = x[:,:,:lf]
    return x_lf

