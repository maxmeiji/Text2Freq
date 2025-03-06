all_models=("PatchTST" "iTransformer" "DLinear" "LightTS") 
start_index=$1
end_index=$2
models=("${all_models[@]:$start_index:$end_index-$start_index+1}")
root_paths=("./data/SocialGood")
data_paths=("SocialGood.csv") 
pred_lengths=(12)
seeds=(2021)
use_fullmodel=0
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for model_name in "${models[@]}"
  do
    for ((i=0; i<$length; i++))
    do
      for pred_len in "${pred_lengths[@]}"
      do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        model_id=$(basename ${root_path})

        echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path \
          --model_id ${model_id}_${seed}_12_${pred_len}_fullLLM_${use_fullmodel} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 12 \
          --label_len 4 \
          --pred_len $pred_len \
          --des 'Exp' \
          --seed $seed \
          --type_tag "#F#" \
          --text_len 4 \
          --prompt_weight 0.1 \
          --pool_type "avg" \
          --save_name "social" \
          --llm_model BERT \
          --huggingface_token 'NA'\
          --use_fullmodel $use_fullmodel \
          --pure_forecast 0 \
          --use_text2freq 1 \
          --use_closedllm 1 \
          --learning_rate_fusion 0.001 \
          --patience 5 \
          --use_freq 1 \
          --lf 3
      done
    done
  done
done

