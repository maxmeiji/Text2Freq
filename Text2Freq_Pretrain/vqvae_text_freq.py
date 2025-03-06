import torch
import torch.nn as nn
from tqdm import tqdm
from model.VQVAE_freq import VQVAE_freq
from data_provider.data_loader_llm import create_data_loader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from model.Trans import Transformer
from torch.optim import AdamW
from utils.freq_utils import time_to_freq, freq_to_time, padding_for_lf
import argparse
from utils.tools import EarlyStopping, crps, DTW
import random
import numpy as np

def train_model(model, vq_model, train_loader, val_loader, test_loader, optimizer, device, num_epochs, best_model_path, lf, patience):
    model.train()
    vq_model.eval()
    best_val_loss = float('inf')
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_rec_loss = 0.0
        train_rec_loss_lf = 0.0
        model.train()
        #train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        train_loader_tqdm = train_loader
        for text, series in train_loader_tqdm:
            series = series.to(device)

            # from series to frequency domain
            series_freq = time_to_freq(series)
            series_freq = padding_for_lf(series_freq, lf)
            optimizer.zero_grad()

            # Get latent space from VQVAE encoder
            with torch.no_grad():
                target_latent = vq_model.encoder(series_freq)
                target_latent = vq_model.quantize(target_latent)[0]  # quantized output
            
            # AR model prediction
            ar_latent_pred = model(text)
            output_series = vq_model.decode(ar_latent_pred)

            # from frequency domain to series
            recon_series = freq_to_time(output_series)
            series_lf = freq_to_time(series_freq)

            rec_loss_lf = criterion(recon_series,series_lf)
            rec_loss = criterion(recon_series,series)

            # Compute MSE loss
            loss = criterion(ar_latent_pred, target_latent)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_rec_loss_lf += rec_loss_lf.item()
            train_rec_loss += rec_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_rec_loss = train_rec_loss / len(train_loader)
        avg_train_rec_loss_lf = train_rec_loss_lf / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss/len(train_loader)}, REC_LF: {train_rec_loss_lf/len(train_loader)}, REC: {train_rec_loss/len(train_loader)}")
        

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_rec_loss = 0.0
        val_rec_loss_lf = 0.0

        # val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
        val_loader_tqdm = val_loader
        with torch.no_grad():
            for text,series in val_loader_tqdm:
                series = series.to(device)
                series_freq = time_to_freq(series)
                series_freq = padding_for_lf(series_freq, lf)

                target_latent = vq_model.encoder(series_freq)
                target_latent = vq_model.quantize(target_latent)[0] 
                ar_latent_pred = model(text)
                output_series = vq_model.decode(ar_latent_pred)

                recon_series = freq_to_time(output_series)
                series_lf = freq_to_time(series_freq)

                rec_loss = criterion(recon_series,series)
                rec_loss_lf = criterion(recon_series, series_lf)

                loss = criterion(ar_latent_pred, target_latent)

                val_loss += loss.item()
                val_rec_loss += rec_loss.item()
                val_rec_loss_lf += rec_loss_lf.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_rec_loss = val_rec_loss / len(val_loader)
            avg_val_rec_loss_lf = val_rec_loss_lf / len(val_loader)

            print(f"[Validation] Loss: {avg_val_loss}, REC Loss: {avg_val_rec_loss}, REC Loss LF: {avg_val_rec_loss_lf}")

            
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with validation loss: {best_val_loss}")
            
            early_stopping(avg_val_loss)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # Testing loop
        model.eval()
        test_loss = 0.0
        test_rec_loss = 0.0
        test_rec_loss_lf = 0.0
        # val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
        test_loader_tqdm = test_loader
        with torch.no_grad():
            for text,series in test_loader_tqdm:
                series = series.to(device)
                series_freq = time_to_freq(series)
                series_freq = padding_for_lf(series_freq, lf)

                target_latent = vq_model.encoder(series_freq)
                target_latent = vq_model.quantize(target_latent)[0] 
                ar_latent_pred = model(text)
                output_series = vq_model.decode(ar_latent_pred)

                recon_series = freq_to_time(output_series)
                series_lf = freq_to_time(series_freq)

                rec_loss = criterion(recon_series,series)
                rec_loss_lf = criterion(recon_series, series_lf)

                loss = criterion(ar_latent_pred, target_latent)

                test_loss += loss.item()
                test_rec_loss += rec_loss.item()
                test_rec_loss_lf += rec_loss_lf.item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_rec_loss = test_rec_loss / len(test_loader)
            avg_test_rec_loss_lf = test_rec_loss_lf / len(test_loader)

            print(f"[TEST] Loss: {avg_test_loss}, REC Loss: {avg_test_rec_loss}, REC Loss LF: {avg_test_rec_loss_lf}")
            
def evaluate(model, vq_model, test_loader, device, best_model_path, pic_save_path):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    vq_model.eval()
    criterion = nn.MSELoss()
    criterion_MAE = nn.L1Loss()
    test_loss = 0.0
    dtw_score = 0.0
    mae_loss = 0.0
    all_outputs = []
    all_series = [] 
    all_texts = []
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating", leave=False)
    test_loader_tqdm = test_loader
    with torch.no_grad():
        for text, series in test_loader_tqdm:
            series = series.to(device)
            
            ar_latent_pred = model(text)
            output_series = vq_model.decode(ar_latent_pred)
            recon_series = freq_to_time(output_series)
            rec_loss = criterion(recon_series,series)
            test_loss += rec_loss.item()
            
            # other metric 
            dtw_score += DTW(series, recon_series)
            mae_loss += criterion_MAE(recon_series,series).item()

            all_outputs.append(recon_series.cpu())
            all_series.append(series.cpu())
            all_texts.extend(text)
        avg_test_loss = test_loss / len(test_loader)
        avg_dtw_score = dtw_score / len(test_loader)
        avg_mae_loss = mae_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss}")
        print(f"Test DTW: {avg_dtw_score}")
        print(f"Test MAE: {avg_mae_loss}")

        # Concatenate all outputs and series for plotting
        all_outputs = torch.cat(all_outputs)
        all_series = torch.cat(all_series)
        # Plot and save predictions for the first 20 data instances
        for i in range(len(all_outputs)):
            plt.figure(figsize=(12, 6))
            plt.plot(all_series[i].numpy(), label='Ground Truth', color='blue')
            plt.plot(all_outputs[i].numpy(), label='Prediction', color='red', linestyle='--')
            plt.title(f"Sample {i + 1} - MSE: {criterion(all_outputs[i].unsqueeze(0), all_series[i].unsqueeze(0)).item():.4f} / CRPS: {crps(all_outputs[i].unsqueeze(0), all_series[i].unsqueeze(0)).item():.4f}")
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
            # Add caption as subtitle or annotate the plot
            plt.suptitle(f"Caption: {all_texts[i]}", fontsize=10)  # Adds the caption text
            
            plt.savefig(os.path.join(pic_save_path, f'sample_{i + 1}.png'))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Transformer Encoder mdoel")
    parser.add_argument('--lf', type=int, default=3, help='Low frequency cutoff value')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--layers', type=int, default=19, help='layers')
    parser.add_argument('--save_path', type=int, default='1', help='transformer save path')
    parser.add_argument('--patience', type=int, default='3', help='transformer save path')
    parser.add_argument('--file', type=str, default='Combine', help='file name')
    parser.add_argument('--series_emb', type=int, default=6, help='the length of embedding sapce')
    parser.add_argument('--series_length', type=int, default=12, help='the length of series')
    parser.add_argument('--caption_length', type=int, default=32, help='the length of caption')
    parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dimensionn')
    parser.add_argument('--stride', type=int, default=1, help='2 for inputs = 48 and 1 for inputs = 12')
    parser.add_argument('--seed', type=int, default=2024, help='random seeds')
    args = parser.parse_args()

    # Hyperparameters
    data_root = './processed_data/Pretraining'
    file = args.file
    file_name = file
    batch_size = 16 
    hidden_dim = args.hidden_dim  
    series_length = args.series_length
    series_emb = args.series_emb
    stride = args.stride
    n_embed = 32  
    embed_dim = 16 
    num_epochs = 800
    learning_rate = 1e-4
    caption_length = args.caption_length
    lf = args.lf
    lr = args.lr
    layers = args.layers
    device = args.device
    save_path = args.save_path
    patience = args.patience
    print(f'lf : {lf}')
    print(f'learning rate: {lr}')
    print(f' device: {device}')


    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    vq_best_path = f"./model_path/vqvae/{file_name}/best_vqvae_model_freq_{lf}.pth"
    best_model_path = f"./model_path/pretrain/{file_name}/pretrain_text_F{lf}.pth"

    pic_save_path = f'./pic/vqvae_text_freq_{lf}/'

    # Load data
    train_loader = create_data_loader(data_root, file, "train", batch_size, shuffle=True)
    val_loader = create_data_loader(data_root, file, "val", batch_size, shuffle=False)
    test_loader = create_data_loader(data_root, file , "test", batch_size,shuffle=False)
    print(len(train_loader), len(val_loader), len(test_loader))


    # Initialize the vqmodel, load the model we trained in the first stage
    vq_model = VQVAE_freq(hidden_dim, series_length, n_embed, embed_dim, stride).to(device)
    vq_model.load_state_dict(torch.load(vq_best_path))
    vq_model.eval()

    # Initialize the transformer model and Train it
    ar_model = Transformer(text_embed_dim=512, n_layers=layers, n_heads=8, hidden_dim=hidden_dim, caption_length = caption_length, series_length=series_emb, device=device).to(device)
    optimizer = AdamW(ar_model.parameters(), lr=lr)
    train_model(ar_model, vq_model, train_loader, val_loader, test_loader, optimizer, device, num_epochs, best_model_path, lf, patience)
    

    # Evaluate the model
    evaluate(ar_model, vq_model, test_loader, device, best_model_path, pic_save_path)


if __name__ == "__main__":
    main()
