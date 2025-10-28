"""
drop audio cond in dataset.py 
unify the text_drop and audio_drop
"""
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import re
from typing import Dict, Optional, Sequence, List, Iterable
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from transformers import WhisperFeatureExtractor, WhisperModel
import tokenizers
from tqdm import tqdm
import bisect
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as Dataset_
from datasets import concatenate_datasets
from univoice.util.utils_smollm import MelSpec, make_pad_mask, MelSpec_bigvGAN

from univoice.constants import *


class TTSDataset(Dataset):
    def __init__(self,data_path: str,
                 tokenizer,tokenizer_fn,
                 data_args):
        super(TTSDataset, self).__init__()

        data = Dataset_.from_file(f"{data_path}/raw.arrow") # single data
            
        with open(f"{data_path}/duration.json", "r", encoding="utf-8") as f:
            duration_dict = json.load(f)
        durations = duration_dict["duration"]
        
        self.tokenizer = tokenizer
        self.data = data
        self.data_args = data_args
        self.data_path = data_path
        self.durations = durations
        self.tokenizer_fn = tokenizer_fn
        
        # MelSpec / MelSpec_bigvGAN
        self.mel_spectrogram = MelSpec_bigvGAN(
            n_fft=data_args.n_fft,
            hop_length=data_args.hop_length,
            win_length=data_args.win_length,
            n_mel_channels=data_args.n_mel_channels,
            target_sample_rate=data_args.target_sample_rate,
            fmax=8000,
        )


    def __len__(self):
        return len(self.data)

    def get_frame_len(self, index):
        if self.durations is not None:
            return self.durations[index] * self.data_args.target_sample_rate / self.data_args.hop_length

    @property
    def lengths(self):
        return self.durations
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"].strip()
            duration = row["duration"]
            # filter by given length
            if 0.3 <= duration <= 30:
                break
            index = (index + 1) % len(self.data)
        audio, source_sample_rate = torchaudio.load(audio_path)
        # mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # resample
        if source_sample_rate != self.data_args.target_sample_rate:
            resampler = torchaudio.transforms.Resample(source_sample_rate, self.data_args.target_sample_rate)
            audio = resampler(audio)
        # mel
        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0) # (D,T)
        # text tokenizer
        conversation = {
            'input': text,
            'output': DEFAULT_SPEECH_TOKEN + '\n',
        }
        
        conversation['input'] = conversation['input'].replace(DEFAULT_SPEECH_TOKEN, f'{DEFAULT_SPEECH_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_SPEECH_END_TOKEN}')
        conversation['output'] = conversation['output'].replace(DEFAULT_SPEECH_TOKEN, f'{DEFAULT_SPEECH_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_SPEECH_END_TOKEN}')


        conversation = [
                {"role": "user", "content": conversation["input"]},
                {"role": "assistant", "content": conversation["output"]},
        ]
        input_ids, labels = self.tokenizer_fn(
                conversation,
                tokenizer=self.tokenizer, 
                source_max_len=512, 
                target_max_len=512, )
   
        data_dict = {  
            "input_ids": input_ids,
            "labels": labels,
            "mel_spec": mel_spec,
            "speechs": torch.rand((128,3000)),  
            "flags": 0,
            "target_len": mel_spec.shape[1],
            "text": text,
        }
        return data_dict


class ASRDataset(Dataset):
    def __init__(self,data_path: str,
                 tokenizer,tokenizer_fn,
                 data_args):
        super(ASRDataset, self).__init__()

        
        data = Dataset_.from_file(f"{data_path}/raw.arrow") # single data
        with open(f"{data_path}/duration.json", "r", encoding="utf-8") as f:
            duration_dict = json.load(f)
        durations = duration_dict["duration"]
        
        self.tokenizer = tokenizer
        self.data = data
        self.data_path = data_path
        self.data_args = data_args
        self.durations = durations
        self.tokenizer_fn = tokenizer_fn
        self.mel_spectrogram = MelSpec_bigvGAN(
            n_fft=data_args.n_fft,
            hop_length=data_args.hop_length,
            win_length=data_args.win_length,
            n_mel_channels=data_args.n_mel_channels,
            target_sample_rate=data_args.target_sample_rate,
            fmax=8000,
        )
        
        self.feature_extracter = WhisperFeatureExtractor.from_pretrained('ckpts/whisper-large-v3-turbo')


    def __len__(self):
        return len(self.data)

    def get_frame_len(self, index):
        if self.durations is not None:
            return self.durations[index] * self.data_args.target_sample_rate / self.data_args.hop_length

    
    @property
    def lengths(self):
        return self.durations
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            # text = row["text"].strip()
            text = row["text"].lower().strip()
            text = re.sub(r"[^a-zA-Z ]", "", text)
            duration = row["duration"]

            if 0.3 <= duration <= 30:
                break

            index = (index + 1) % len(self.data)
        
        audio, source_sample_rate = torchaudio.load(audio_path)
        # mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # resample
        if source_sample_rate != 16000:   # whisper---16KHZ
            resampler = torchaudio.transforms.Resample(source_sample_rate, 16000)
            audio = resampler(audio)

        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0) # (D,T)


        speech_inputs = self.feature_extracter(audio.numpy(), sampling_rate=16000).input_features[0]
        speech_inputs = torch.tensor(speech_inputs, dtype=torch.float32)

        # text tokenizer
        conversation = {
            'input': f'{DEFAULT_SPEECH_TOKEN}' + '<|asr|>',
            'output': f"{text}", 
        }
        conversation = [
            {"role": "user", "content": conversation["input"]},
            {"role": "assistant", "content": conversation["output"]},
        ]
        input_ids, labels = self.tokenizer_fn(
            conversation,
            tokenizer=self.tokenizer, 
            source_max_len=512, 
            target_max_len=512, )

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "mel_spec": mel_spec, 
            "speechs": speech_inputs,
            "flags": 1,
            "target_len": speech_inputs.shape[1],
            "text": text,
        }
        return data_dict


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes

    def get_frame_len(self, index):
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cummulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_frame_len(sample_idx)

   
def tokenize_conversation(conversation, tokenizer, source_max_len, target_max_len):
    sources = []
    targets = []
    sources = [f"{tokenizer.bos_token}{conv['content']}" for conv in conversation if conv["role"] == "user"] # |im_start| 151644
    targets = [f"{conv['content']}{tokenizer.eos_token}" for conv in conversation if conv["role"] == "assistant"]
    
    # Tokenize
    sources = tokenizer(
        sources,
        max_length=source_max_len,
        truncation=True,
        add_special_tokens=False,
    )
    targets = tokenizer(
        targets,
        max_length=target_max_len,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = []
    labels = []

    for source_ids, target_ids in zip(sources['input_ids'], targets['input_ids']):
        input_ids.append(torch.tensor(source_ids + target_ids))
        labels.append(torch.tensor(copy.deepcopy(source_ids + target_ids)))
    
    input_ids = torch.cat(input_ids)
    labels = torch.cat(labels)

    im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    
    for idx in torch.where(input_ids == im_start_token_id):
        if len(idx) != 0 and input_ids[idx + 1] == tokenizer.pad_token_id:
            input_ids[idx + 1] = speech_token_id
    
    return input_ids, labels


@dataclass
class DataCollatorForSupervisedDataset(object):
    """single-task data collator."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # phone_ids, phone_mask, mel, mel_mask, task
        input_ids = [x['input_ids'] for x in instances]
        labels = [x['labels'] for x in instances]
        mel_spec = [x['mel_spec'].transpose(0,1) for x in instances]
        speechs = [x['speechs'].transpose(0,1) for x in instances]
        flags = [x['flags'] for x in instances]
        target_len = [x['target_len'] for x in instances]
        text = [x['text'] for x in instances]
        # task = instances[0][6]
        # =text
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels
        }

        mel_spec = pad_sequence(mel_spec, batch_first=True, padding_value=0)
        speechs = pad_sequence(speechs, batch_first=True, padding_value=0)
        flags = torch.tensor(flags, dtype=torch.int32)
        target_len = torch.tensor(target_len, dtype=torch.int32)

        data_dict['mel_spec'] = mel_spec # for generation
        data_dict['speechs'] = speechs  # for understading
        data_dict['flags'] = flags  # indicator
        data_dict['target_len'] = target_len
        data_dict['text'] = text
        # data_dict['task'] = task

        return data_dict




def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.task == 'tts':
        train_dataset = TTSDataset(tokenizer=tokenizer,data_path=data_args.tts_train_data_path,tokenizer_fn=tokenize_conversation,
                                data_args=data_args)
        eval_dataset = TTSDataset(tokenizer=tokenizer,data_path=data_args.eval_data_path,tokenizer_fn=tokenize_conversation,
                                data_args=data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.task == 'asr':
        train_dataset = ASRDataset(tokenizer=tokenizer,data_path=data_args.asr_train_data_path,tokenizer_fn=tokenize_conversation,
                                data_args=data_args)
        eval_dataset = ASRDataset(tokenizer=tokenizer,data_path=data_args.eval_data_path,tokenizer_fn=tokenize_conversation,
                                data_args=data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif data_args.task == 'all':
        train_dataset_tts = TTSDataset(tokenizer=tokenizer,data_path=data_args.tts_train_data_path,tokenizer_fn=tokenize_conversation,
                                    data_args=data_args)
        train_dataset_asr = ASRDataset(tokenizer=tokenizer,data_path=data_args.asr_train_data_path,tokenizer_fn=tokenize_conversation,
                                    data_args=data_args)
        train_dataset = ConcatDataset([train_dataset_tts,train_dataset_asr])
        
        eval_dataset = TTSDataset(tokenizer=tokenizer,data_path=data_args.eval_data_path,tokenizer_fn=tokenize_conversation,
                                    data_args=data_args)

        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
