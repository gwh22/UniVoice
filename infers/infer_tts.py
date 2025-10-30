import argparse
import json
import multiprocessing as mp
import os
import socket
from typing import List, Optional

import transformers
import random
import numpy as np
import torch
import torchaudio
import torch.distributed as dist
from transformers import AutoTokenizer

from transformers import logging
logging.set_verbosity_error()

from univoice.model import *
from univoice.constants import *

from univoice.util.utils_smollm import MelSpec, make_pad_mask, MelSpec_bigvGAN
from univoice.tensor_util import spec_to_figure, spec_to_figure_single
from univoice.train.train_utils import set_all_random_seed, setup_tokenizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--cfg_scale", type=float, required=True)
    parser.add_argument("--dur", type=float, required=True)
    
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    setup_seed(42) # random seed default=42

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    model = UniVoiceForCausalLM.from_pretrained(
        args.llm_path, torch_dtype=torch.float32
    )
    model, tokenizer = setup_tokenizer(model, tokenizer)
    model.config.speech_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    
    checkpoint = torch.load(f'{args.ckpt_path}/model_last.pt', map_location='cuda', weights_only=True)
    
    model.load_state_dict(checkpoint["model_state_dict"],strict=False)

    del checkpoint
    torch.cuda.empty_cache()


    

    wav_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/train-clean-100/103/1241/103-1241-0001.flac"

    
    audio, source_sample_rate = torchaudio.load(wav_path)

    if audio.shape[0] > 1: # mono
        audio = torch.mean(audio, dim=0, keepdim=True)
    if source_sample_rate != 22050:   # whisper---16KHZ
        resampler = torchaudio.transforms.Resample(source_sample_rate, 22050)
        audio = resampler(audio)

    mel_spec = MelSpec_bigvGAN(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        target_sample_rate=22050,
    )(audio).to('cuda')
    dur = args.dur
    speechs = mel_spec
    flags = torch.tensor([0])

    target_len = torch.tensor([int(dur*22050//256)])  

    text = ["OR A HOLLOW WHERE WILD PLUMS HUNG OUT THEIR FILMY BLOOM THE AIR WAS SWEET WITH THE BREATH OF MANY APPLE ORCHARDS AND THE MEADOWS SLOPED AWAY IN THE DISTANCE TO HORIZON MISTS OF PEARL AND PURPLE, WHILE THE LITTLE BIRDS SANG AS IF IT WERE THE ONE DAY OF SUMMER IN ALL THE YEAR".lower()]
    temp = torch.randn(1).to('cuda')
    with torch.no_grad():
        mel_out, mel_gt = model.sample(
            input_ids=temp,
            attention_mask=temp,
            labels=temp,
            mel_spec=mel_spec.transpose(1,2),
            speechs=speechs,
            flags=flags,
            target_len=target_len,
            text=text,
            cfg_scale=args.cfg_scale
        )
    print('mel_out:',mel_out.shape)
    os.makedirs('infers/', exist_ok=True)
    spec_to_figure(mel_out, title="", file_name=f"infers/pred_out.png")
    spec_to_figure(mel_gt, title="", file_name=f"infers/gt.png")
    # bigvagn vocoder
    from BigVGAN import bigvgan
    vocoder = bigvgan.BigVGAN.from_pretrained(
        '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/gwh/Proj/bigvgan_22k', 
        use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to('cuda')

    # generate waveform from mel
    with torch.inference_mode():
        wav_gen = vocoder(mel_out.squeeze(0).transpose(0,1).unsqueeze(0)) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
        wav_gt = vocoder(mel_gt.squeeze(0).transpose(0,1).unsqueeze(0))
    wav_gen_float = wav_gen.squeeze(0).cpu()
    wav_gt_float = wav_gt.squeeze(0).cpu()

    torchaudio.save(f'infers/out.wav', wav_gen_float, 22050)
    torchaudio.save(f'infers/gt.wav', wav_gt_float, 22050)


if __name__ == "__main__":
    main()

