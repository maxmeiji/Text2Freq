import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
from transformers import BertTokenizer, BertModel


class Transformer(nn.Module):
    def __init__(self, text_embed_dim=512, n_layers=19, n_heads=16, hidden_dim=64, caption_length = 64, series_length=12, device = 'cuda'):
        super(Transformer, self).__init__()
        self.device = device
        self.hidden = hidden_dim
        self.caption_length = caption_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embedding = nn.Linear(self.bert.config.hidden_size , text_embed_dim)
        encoder_layers = TransformerEncoderLayer(d_model=text_embed_dim, nhead=n_heads, dropout=0.3)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(text_embed_dim, hidden_dim)
        self.out_proj = nn.Linear(text_embed_dim, hidden_dim)  # Mapping to VQVAE latent space
        self.sequence_projector = nn.Linear(caption_length, series_length)
        self.dropout = nn.Dropout(0.3)
    def forward(self, text):
        self.bert.eval()
        text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.caption_length, add_special_tokens=True)
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        # Extract BERT embeddings
        text_embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # print(text_embeddings.shape) [batch_size, padding_seq_length, dimension]
        # Pass through BERT's pooler output
        text_embeddings = text_embeddings.to(next(self.parameters()).device)
        # Print the shape of text embeddings (should be [batch_size, latent shape, 768])
        # print(text_embeddings.shape)  # [batch_size, 64, 768]
        embedded_text = self.embedding(text_embeddings)  
        #print(embedded_text.shape) [16,hidden=64,512]
        transformer_out = self.transformer(embedded_text) 
        #print(transformer_out.shape) [126,hidden=64,512]
        output_hidden = self.out_proj(transformer_out)  
        output_hidden = self.dropout(output_hidden)
        #print(output_hidden.shape) [16,64,32] 
        output = self.sequence_projector(output_hidden.transpose(1, 2))
        output = self.dropout(output)
        # print(output.shape) 
        # print(output.shape) [16,32,12]
        return output

    

