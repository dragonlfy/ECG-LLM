model_name=FEDformer

# training one model with a context length
python -u run_ecg.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ECG_CSV_less/ \
  --data_path merged_data_less.csv \
  --model_id ECG_576_512 \
  --model $model_name \
  --data ECG \
  --seq_len 640 \
  --label_len 512 \
  --token_len 128 \
  --test_seq_len 640 \
  --test_label_len 128 \
  --test_pred_len 512 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 20 \
  --use_amp \
  --gpu 2\
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last