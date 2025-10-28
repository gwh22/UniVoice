export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 36507 --nproc_per_node 1 infer_small_f5_ema/infer_asr.py \
    --ckpt_path /ckpts/univoice_all \
    --llm_path /ckpts/SmolLM2-360M \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k 50 \
    --num_beams 4


