import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, hidden_dim, n_res_blocks=2, input_length=12, stride = 2):
        super(Encoder, self).__init__()
        # Initial convolution to expand feature dimensions
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=stride+2, stride=stride, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim, n_res_blocks=2, input_length=12, stride=2):
        super(Decoder, self).__init__()
        # Residual blocks for decoding
        self.deconv1 = nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=stride+2, stride=stride, padding=1)
        self.relu2 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv4(x)
        x = x.squeeze(1)
        return x