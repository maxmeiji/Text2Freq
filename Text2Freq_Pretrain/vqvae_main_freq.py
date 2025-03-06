import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model.VQVAE_freq import VQVAE_freq
from data_provider.data_loader_llm import create_data_loader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from utils.freq_utils import time_to_freq, freq_to_time, padding_for_lf
import argparse

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, best_model_path, lf):
    best_val_loss = float('inf')
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        rec_train_loss = 0.0

        #train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        train_loader_tqdm = train_loader
        for _,series in train_loader_tqdm:
            series = series.to(device)
        
            # from series to frequency domain
            series_freq = time_to_freq(series)
            # padding for low frequency
            series_freq = padding_for_lf(series_freq, lf)
            optimizer.zero_grad()
            reconstructed_series, diff = model(series_freq)

            # from frequency domain to series
            recon_series = freq_to_time(reconstructed_series)
            series_lf = freq_to_time(series_freq)

            loss, recon_loss = model.loss_function(recon_series, series_lf, diff)
            loss.backward()

            rec_loss = F.mse_loss(recon_series, series, reduction='mean')
            train_loss += loss.item()
            rec_train_loss += rec_loss.item()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss/len(train_loader)}, Rec: {rec_train_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        val_rec_loss = 0.0
        val_rec_total_loss = 0.0
        #val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
        val_loader_tqdm = val_loader
        with torch.no_grad():
            for _,series in val_loader_tqdm:
                series = series.to(device)
                series_freq = time_to_freq(series)
                series_freq = padding_for_lf(series_freq, lf)

                reconstructed_series, diff = model(series_freq)

                recon_series = freq_to_time(reconstructed_series)
                series_lf = freq_to_time(series_freq)

                recon_loss = criterion(recon_series, series_lf)
                rec_loss = criterion(recon_series, series)

                val_rec_loss += recon_loss
                val_rec_total_loss += rec_loss

            avg_val_loss = val_rec_loss / len(val_loader)
            avg_rec_total = val_rec_total_loss / len(val_loader)
            print(f"[Validation] Loss: {avg_val_loss}, Rec: {avg_rec_total}")

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with validation loss: {best_val_loss}")

def evaluate(model, test_loader, device, best_model_path, pic_save_path, lf):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_outputs = []
    all_series = [] 
    all_texts = []
    # test_loader_tqdm = tqdm(test_loader, desc="Evaluating", leave=False)
    test_loader_tqdm = test_loader
    with torch.no_grad():
        for text,series in test_loader_tqdm:
            series = series.to(device)
            series_freq = time_to_freq(series)
            series_freq = padding_for_lf(series_freq, lf)
            reconstructed_series, diff = model(series_freq)
            recon_series = freq_to_time(reconstructed_series)
            recon_loss = criterion(recon_series, series)
            test_loss += recon_loss
            all_outputs.append(recon_series.cpu())
            all_series.append(series.cpu())
            all_texts.extend(text)
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss}")

        # Concatenate all outputs and series for plotting
        all_outputs = torch.cat(all_outputs)
        all_series = torch.cat(all_series)
        # Plot and save predictions for the first 20 data instances
        for i in range(min(20, len(all_outputs))):
            plt.figure(figsize=(12, 6))
            plt.plot(all_series[i].numpy(), label='Ground Truth', color='blue')
            plt.plot(all_outputs[i].numpy(), label='Prediction', color='red', linestyle='--')
            plt.title(f"Sample {i + 1} - MSE: {criterion(all_outputs[i].unsqueeze(0), all_series[i].unsqueeze(0)).item():.4f}")
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
            # Add caption as subtitle or annotate the plot
            plt.suptitle(f"Caption: {all_texts[i]}", fontsize=10)  # Adds the caption text
            
            plt.savefig(os.path.join(pic_save_path, f'sample_{i + 1}.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train VQVAE model with text and frequency domain input")
    parser.add_argument('--lf', type=int, default=6, help='Low frequency cutoff value')
    parser.add_argument('--file', type=str, default='Combine', help='file_name')
    parser.add_argument('--hidden_dim', type=int, default=16, help='hidden_dim')
    parser.add_argument('--series_length', type=int, default=12, help='series length')
    parser.add_argument('--n_embed', type=int, default=32, help='number of VQVAE embedding space')
    parser.add_argument('--embed_dim', type=int, default=16, help='embedding dimension')
    parser.add_argument('--stride', type=int, default=1, help='2 for inputs = 48 and 1 for inputs = 12 ')
    args = parser.parse_args()

    # Hyperparameters
    data_root = './processed_data/Pretraining'
    file = args.file
    file_name = file
    batch_size = 16
    hidden_dim = args.hidden_dim 
    series_length = args.series_length
    n_embed = args.n_embed 
    embed_dim = args.embed_dim 
    stride = args.stride
    num_epochs = 250
    learning_rate = 1e-3
    lf = args.lf

    print(lf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    best_model_path = f"./model_path/vqvae/{file_name}/best_vqvae_model_freq_{lf}.pth"
    pic_save_path = './pic/vqvae/'

    # Load data
    train_loader = create_data_loader(data_root, file, "train", batch_size, shuffle=True) 
    val_loader = create_data_loader(data_root, file, "val", batch_size,shuffle=False) 
    test_loader = create_data_loader(data_root, file, "test", batch_size,shuffle=False) 

    # Initialize model, optimizer
    model = VQVAE_freq(hidden_dim, series_length, n_embed, embed_dim, stride).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # Train and validate the model
    train_model(model, train_loader, val_loader, optimizer, device, num_epochs, best_model_path, lf)
    
    # Evaluate the model
    evaluate(model, test_loader, device, best_model_path, pic_save_path, lf)

if __name__ == "__main__":
    main()

