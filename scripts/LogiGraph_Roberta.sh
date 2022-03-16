#!/usr/bin/env bash
export DATASET_DIR=ReclorDataset
export TASK_NAME=LogiGraph
export MODEL_DIR=$1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

export RUN_NAME=AdaLoGN_Reclor

export DATASET_DIR=$DATASET_DIR
export MODEL_TYPE=Roberta

CUDA_VISIBLE_DEVICES=0 python run_multiple_choice.py \
  --run_name $RUN_NAME \
  --task_name $TASK_NAME \
  --model_name_or_path $MODEL_DIR \
  --data_dir $DATASET_DIR \
  --$MODE \
  --do_eval \
  --seed 123 \
  --model_type $MODEL_TYPE \
  --max_seq_length 384 \
  --per_device_eval_batch_size 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 10 \
  --output_dir Checkpoints/$DATASET_DIR/$RUN_NAME \
  --logging_steps 200 \
  --learning_rate 7e-6 \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model acc_dev \
  --gnn_layers_num 2 \
  --save_total_limit 2 \
  --dropout 0.1 \
  --warmup_ratio 0.1 \
  --pooling_type attention_pooling_with_gru
