export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 36505 --nproc_per_node 1 infer_small_f5_ema/infer_tts_single_bigvgan.py \
    --ckpt_path  /ckpts/univoice_all \
    --llm_path /ckpts/SmolLM2-360M \
    --cfg_scale 2 \
    --dur 17

