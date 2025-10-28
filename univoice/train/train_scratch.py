import logging
import os
import pathlib
import random
import json
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
from torch.utils.data import Sampler, SequentialSampler, DataLoader, Dataset
import datasets
from datetime import datetime
import sys
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
import socket
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import set_seed
from tqdm import tqdm
from time import time

from univoice.model import *
from univoice.constants import *
from univoice.train.univoice_trainer_scratch import Trainer

from univoice.dataset_smollm import make_supervised_data_module, TTSDataset, ASRDataset, ConcatDataset,\
     tokenize_conversation, DataCollatorForSupervisedDataset, make_supervised_data_module
from univoice.tensor_util import spec_to_figure, spec_to_figure_single
from univoice.train.train_utils import set_all_random_seed, setup_tokenizer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    # model_type: Optional[str] = field(default=None)
    cfg_scale: int = field(default=1) # 3 for condition implementation

@dataclass
class DataArguments:
    tts_train_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    asr_train_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str =field(default=None,
                           metadata={"help": "Path to the eval data."})
    target_sample_rate: int = field(default=24000)
    hop_length: int = field(default=256)
    n_fft: int = field(default=1024)
    win_length: int = field(default=1024)
    n_mel_channels: int = field(default=100)
    batch_size_per_gpu: int = field(default=38400) # per gpu
    max_samples: int = field(default=64) # per gpu
    task: str =field(default='all', metadata={"help": "Path to the eval data."})
    tts_ratio: int = field(default=1)
    prompt_path: str =field(default=None)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )   
    output_dir: str = field(default="output_dir")
    max_steps: int = field(default=100000)
    num_workers: int = field(default=4)
    mixed_precision: str = field(default='bf16')
    lr: float = field(default=1e-4)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0)
    seed: int = field(default=11)
    allow_tf32: bool = True
    resume_from_checkpoint: str = field(default="")
    init_from_checkpoint: str = field(default="")
    checkpoints_total_limit: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    eval_batch_size: int = field(default=1)
    log_steps: int = field(default=1)
    save_per_updates: int = field(default=10000)
    last_per_updates: int = field(default=10000)
    keep_last_n_checkpoints: int = field(default=1)
    log_level: str = field(default="INFO")
    logdir: str = field(default="logs")
    epochs: int = field(default=1)
    warmup_iters: int = field(default=10000)
    lr_decay_iters: int = field(default=10000)
    gradient_checkpointing: bool = False
    lr_decay_rate: float = field(default=0.1)
    use_lr_decay: bool = True
    batch_size_type: str = field(default="frame") # sample
    logger: str = field(default="tensorboard") 
    log_samples: bool = True





def train():
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        add_bos_token=True, add_eos_token=True
    )
    
    # UniVoiceSmolLMConfig, UniVoiceForCausalLM
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    model = UniVoiceForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch_dtype,)
    model.get_model().initialize_weights()
    model, tokenizer = setup_tokenizer(model, tokenizer)
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    model.config.speech_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print('bos:',tokenizer.bos_token_id,tokenizer.bos_token)
    print('eos:',tokenizer.eos_token_id,tokenizer.eos_token)
    

    
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    training_args.batch_size_per_gpu = data_args.batch_size_per_gpu
    training_args.max_samples = data_args.max_samples

    # init trainer
    trainer = Trainer(
        model,
        epochs=training_args.epochs,
        learning_rate=training_args.lr,
        num_warmup_updates=training_args.warmup_iters,
        save_per_updates=training_args.save_per_updates,
        keep_last_n_checkpoints=training_args.keep_last_n_checkpoints,
        checkpoint_path=training_args.output_dir,
        batch_size_per_gpu=training_args.batch_size_per_gpu,
        batch_size_type=training_args.batch_size_type,
        max_samples=training_args.max_samples,
        grad_accumulation_steps=training_args.gradient_accumulation_steps,
        max_grad_norm=training_args.max_grad_norm,
        logger=training_args.logger,
        last_per_updates=training_args.last_per_updates,
        mixed_precision=training_args.mixed_precision,
        log_samples=training_args.log_samples,
        bnb_optimizer=False,
        collator=data_module['data_collator'],
        weight_decay=training_args.weight_decay,
        tokenizer=tokenizer
    )


    # train_loader = get_train_dataloader(training_args, data_module['train_dataset'], data_module['data_collator'])
    # eval_loader = get_eval_dataloader(training_args, data_module['eval_dataset'], data_module['data_collator'])


    trainer.train(
        data_module['train_dataset'],
        data_module['eval_dataset'],
        num_workers=training_args.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )




if __name__ == "__main__":
    train()

