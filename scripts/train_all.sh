#!/bin/bash
export PYTHONPATH=.
export NCCL_P2P_DISABLE=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL
exp_name='smollm_univoice'
torchrun --nproc_per_node=4  --master_port 25901 univoice/train/train_scratch.py \
    --model_name_or_path hf_ckpts/SmolLM2-360M-Instruct \
    --tts_train_data_path data/libriheavy_train \
    --asr_train_data_path data/libriheavy_train \
    --eval_data_path data/librispeech_pc_eval \
    --task all \
    --bf16 True \
    --output_dir checkpoints/univoice/${exp_name} \
    --epochs 10 \
    --max_steps 1000000 \
    --batch_size_per_gpu 20000 \
    --max_samples 64 \
    --batch_size_type frame \
    --n_mel_channels 80 \
    --target_sample_rate 22050 \
    --gradient_accumulation_steps 1 \
    --num_workers 4 \
    --lr 1e-3 \
    --logger tensorboard \
    --mixed_precision bf16 \
    --max_grad_norm 1 \
    --weight_decay 0.05 \
    --checkpoints_total_limit 100 \
    --eval_batch_size 1 \
    --log_steps 100 \
    --save_per_updates 10000 \
    --last_per_updates 5000 \
    --keep_last_n_checkpoints 50 \
    --log_samples True \
    --logdir checkpoints/univoice/${exp_name}/logs \
    --warmup_iters 100 \
    --lr_decay_iters 500000 \
    --lr_decay_rate 0.1 \
    2>&1 | tee -a checkpoints/univoice/${exp_name}/log.txt && echo "Done."






