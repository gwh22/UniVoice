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

from univoice.train.train_utils import set_all_random_seed, setup_tokenizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], speechs: List[torch.Tensor], max_length=512, device='cuda'):
    """
    Currently, only support batch size 1.
    """
    assert len(inputs) == 1

    input_ids, attention_mask = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).values()
    

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    flags = [1]

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'speechs': speechs[0].transpose(0,1).unsqueeze(0),
        'flags': torch.tensor(flags,dtype=torch.int32),
        't': torch.tensor([0]).to(device),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--llm_path", type=str, required=True)
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
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


    ckpt_type = args.ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(f'{args.ckpt_path}/model_last.pt', device='cuda')
    else:
        checkpoint = torch.load(f'{args.ckpt_path}/model_last.pt', map_location='cuda')
    
    if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}

    model.load_state_dict(checkpoint["model_state_dict"],strict=False)

    del checkpoint
    torch.cuda.empty_cache()

    model.eval().cuda()

    feature_extracter = WhisperFeatureExtractor.from_pretrained('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/gwh/Proj/whisper-large-v3-turbo')

    # asr wav_path
    wav_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean/672/122797/672-122797-0053.flac"
    audio, source_sample_rate = torchaudio.load(wav_path)
    if audio.shape[0] > 1: # mono
        audio = torch.mean(audio, dim=0, keepdim=True)
    if source_sample_rate != 16000:   # whisper---16KHZ
        resampler = torchaudio.transforms.Resample(source_sample_rate, 16000)
        audio = resampler(audio)
    
    mel_spec = feature_extracter(audio.numpy(), sampling_rate=16000).input_features[0]
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    print('mel:',mel_spec.shape) # (80,3000)
    # speechs and prompt
    speechs = [mel_spec.to('cuda')]
    inputs = [f"{tokenizer.bos_token}{DEFAULT_SPEECH_TOKEN}"+"<|asr|>"]

    inputs_dict = preprocess_inputs(tokenizer, inputs, speechs)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs_dict['input_ids'],
            attention_mask=inputs_dict['attention_mask'],
            speechs=inputs_dict['speechs'],
            flags=inputs_dict['flags'],
            t=inputs_dict['t'],
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[1]]
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    output_ids = outputs.replace("\n"," ").replace("<|im_end|>","")
    print(output_ids)


if __name__ == "__main__":
    main()



