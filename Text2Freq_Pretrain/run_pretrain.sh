#!/bin/bash
file="Combine"
hidden_dim=16
series_length=12
stride=1
lr=1e-5
device="cuda"
layers=3
patience=25
series_emb=6
caption_length=32
seed=2023

# Run vqvae model to extract series latent space

# For series domain
# python vqvae_main.py --file $file --hidden_dim $hidden_dim --series_length $series_length --stride $stride > "result_pre0.txt" 2>&1

# For frequency domain
for i in {1..6}
do
    python vqvae_main_freq.py --file $file --hidden_dim $hidden_dim --series_length $series_length --stride $stride --lf $i > "result_pre${i}.txt" 2>&1
done

# Run transformer encoder model to align text to series latent space

# For series domain
# python vqvae_text.py \
#  --lr $lr --device $device --layers $layers --patience $patience \
#  --file $file --series_emb $series_emb --caption_length $caption_length \
#  --hidden_dim $hidden_dim --stride $stride --seed $seed > "seed${seed}_${file}_0.txt" 2>&1

# For frequency domain
for i in {1..6}
do
    python vqvae_text_freq.py \
      --lf $i --lr $lr --device $device --layers $layers --patience $patience \
      --file $file --series_emb $series_emb --caption_length $caption_length \
      --hidden_dim $hidden_dim --stride $stride --seed $seed > "seed${seed}_${file}_${i}.txt" 2>&1
done

# For visualization
# for i in {0..6}
# do 
#     python tsne_color.py --lf $i
# done 