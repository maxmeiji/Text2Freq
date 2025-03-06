import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model.VQVAE import VQVAE
from model.VQVAE_freq import VQVAE_freq
from PIL import Image
from data_provider.data_loader_llm import create_data_loader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from utils.freq_utils import time_to_freq, freq_to_time, padding_for_lf
from transformers import BertTokenizer, BertModel
import argparse
from model.Trans import Transformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Set the random seed for reproducibility
random_seed = 30
torch.manual_seed(random_seed)

def evaluate(ar_model, vq_model, test_loader, lf, device, best_model_path, pic_save_path):
    ar_model.load_state_dict(torch.load(best_model_path))
    ar_model.eval()
    vq_model.eval()
    bert_model.to(device)
    bert_model.eval() 
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating", leave=False)
    
    text_embeddings_list = []
    series_embeddings_list = []
    texts = []
    with torch.no_grad():
        for batch_idx, (text, series) in enumerate(test_loader_tqdm):
            if lf > 0:
                series_freq = time_to_freq(series)
                series = padding_for_lf(series_freq, lf).to(device)
            else:
                series = series.to(device)

            texts.append(text)

            # Process series to get series embeddings
            target_latent = vq_model.encoder(series)
            target_latent = vq_model.quantize(target_latent)[0]  # Shape [b, l, h]
            series_embeddings = target_latent.view(target_latent.shape[0],-1)
            
            text_embeddings = ar_model(text)    
            text_embeddings = text_embeddings.view(text_embeddings.shape[0],-1)
            text_embeddings_list.append(text_embeddings.cpu().detach().numpy())
            series_embeddings_list.append(series_embeddings.cpu().detach().numpy())


    # Concatenate all embeddings (text and serie
    text_embeddings_all = np.concatenate(text_embeddings_list, axis=0)
    series_embeddings_all = np.concatenate(series_embeddings_list, axis=0)
    print(text_embeddings_all.shape)
    print(series_embeddings_all.shape)
    text_cosine_distances = 1 - cosine_similarity(text_embeddings_all)
    series_cosine_distances = 1 - cosine_similarity(series_embeddings_all)
    
    # compute r2_score with two distances 
    ref_index = 53
    x = [text_cosine_distances[ref_index, i]  for i in range(len(text_embeddings_all))]
    y = [series_cosine_distances[ref_index, i]  for i in range(len(series_embeddings_all))]
    # x = [text_cosine_distances[ref_index, i]  for i in range(46)]
    # y = [series_cosine_distances[ref_index, i]  for i in range(46)]
    
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    r2_score = model.score(x, y)
    y_pred = model.predict(x)

    
    Visualize(lf, pic_save_path, x, y, y_pred, r2_score)



def Visualize(lf, pic_save_path, x,y,y_pred, r2_score):
    # The reference point we chose in the datasets
    ref_index = 53

    # All data points plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(x[ref_index], y[ref_index], color = 'black', label = 'Ref.')
    plt.scatter(x[:53], y[:53], color = 'blue', label = 'All data points')
    plt.scatter(x[54:], y[54:], color = 'blue')
    # Specify increasing, fluctuating and decreasing
    #plt.scatter(x[1:16], y[1:16], color = 'darkblue', label = 'increasing')
    #plt.scatter(x[16:31], y[16:31], color = 'darkorange', label='fluctuating')
    #plt.scatter(x[31:], y[31:], color = 'darkgreen', label='decreasing')

    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.legend()

    plt.title(f"$R^2$ score: {round(r2_score, 3)}", fontsize=25)

    plt.xlabel("Cosine Distance (Text)", fontsize=20)
    plt.ylabel("Cosine Distance (Series)", fontsize=20)
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(pic_save_path, f'{lf}_distance.png'))


def main():
    parser = argparse.ArgumentParser(description="Train VQVAE model with text and frequency domain input")
    parser.add_argument('--lf', type=int, default=100, help='Low frequency cutoff value')
    args = parser.parse_args()

    # Hyperparameters
    data_root = './processed_data/Pretraining'
    file = 'Combine_Final'
    test_file = 'Combine_Final'
    batch_size = 1 
    hidden_dim = 16  
    series_length = 12
    series_emb = 6
    stride = 1
    n_embed = 32  
    embed_dim = 16 
    num_epochs = 800
    learning_rate = 1e-4
    caption_length = 32
    lf = args.lf
    print(lf)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if lf > 0: 
        vq_best_path = f"./model_path/vqvae/{file}/best_vqvae_model_freq_{lf}.pth"
        best_model_path = f"./model_path/pretrain/{file}/pretrain_text_F{lf}.pth"
        vq_model = VQVAE_freq(hidden_dim, series_length, n_embed, embed_dim, stride).to(device)
    else: 
        vq_best_path = f"./model_path/vqvae/{file}/best_vqvae_model.pth"
        best_model_path = f"./model_path/pretrain/{file}/pretrain_text_series.pth"
        vq_model = VQVAE(hidden_dim, series_length, n_embed, embed_dim, stride).to(device)
    pic_save_path = f'./pic/2dvisualization/{file}/'

    # Load data
    test_loader = create_data_loader(data_root, test_file, "test", batch_size, shuffle=False)


    # Initialize the vqmodel, load the model we trained in the first stage
    vq_model.load_state_dict(torch.load(vq_best_path))
    vq_model.eval()

    ar_model = Transformer(text_embed_dim=512, n_layers=3, n_heads=8, hidden_dim=hidden_dim, caption_length = caption_length, series_length=series_emb, device=device).to(device)    

    # Evaluate the model
    evaluate(ar_model, vq_model, test_loader, lf, device, best_model_path, pic_save_path)

if __name__ == "__main__":
    main()

