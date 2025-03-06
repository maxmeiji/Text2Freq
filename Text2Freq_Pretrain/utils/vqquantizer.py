import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE.
    Discretizes continuous inputs by mapping them to the closest embeddings.
    
    Args:
        n_e: Number of embeddings. 64
        e_dim: Dimension of each embedding. 32
        beta: Commitment cost used in the loss term.
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
        Args:
            z: Input from the encoder with shape (batch, channels, time).
        
        Returns:
            z_q: Quantized output.
            loss: The VQ loss.
            info: Tuple containing additional information (perplexity, min_encodings, min_encoding_indices).
        """
        # Flatten the input
        # print(z.shape) [b, h_d, s] = [16, 32, 12]
        z_flattened = z.view(-1, self.e_dim)
        # print(z_flattened.shape) [bxs, e_dim] = [192, 32]
        # Compute distances between z and the embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # Get the closest embedding index for each input
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # print(f'min_encodding_indicies: {min_encoding_indices.shape}') [192,1]
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        # print(min_encodings.shape) [192, 64]
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # print(min_encodings.shape) [192, 64]
        # Quantize the latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # print(z_q.shape) [16, 32, 12]
        # Compute the VQ loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Compute perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, loss, min_encoding_indices, self.embedding.weight

    def get_codebook_entry(self, indices):
        # input indices [b, series_length]
        batch_size, series_length = indices.shape[0], indices.shape[1]
        # print(indices.shape)
        indices = indices.view(-1,1).long()
        
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices.device)
        # print(min_encodings.shape)
        min_encodings.scatter_(1, indices, 1)
        # print(min_encodings.shape) [192, 64]
        # Quantize the latent vectors
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(batch_size, self.e_dim, series_length)

        return z_q
