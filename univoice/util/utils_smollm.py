from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchaudio
from typing import Tuple, List
from librosa.filters import mel as librosa_mel_fn
import transformers
from univoice.constants import *
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import json
from importlib import import_module
import inspect

mel_basis_cache = {}
hann_window_cache = {}

# vocos
class MelSpec_hifigan(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=False,
            norm="slaney",
            onesided=True,
            mel_scale="slaney",
        )

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, waveform):
        if self.dummy.device != waveform.device:
            self.to(waveform.device)

        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(waveform.shape) == 2

        mel = self.mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel

class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        )

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, waveform):
        if self.dummy.device != waveform.device:
            self.to(waveform.device)

        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(waveform.shape) == 2

        mel = self.mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel


mel_basis_cache = {}
hann_window_cache = {}
class MelSpec_bigvGAN(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        fmin=0,
        fmax=8000,
        center=False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, waveform):
        if self.dummy.device != waveform.device:
            self.to(waveform.device)

        device = waveform.device
        key = f"{self.n_fft}_{self.n_mel_channels}_{self.target_sample_rate}_{self.hop_length}_{self.win_length}_{self.fmin}_{self.fmax}_{device}"

        if key not in mel_basis_cache:
            mel = librosa_mel_fn(
                sr=self.target_sample_rate, 
                n_fft=self.n_fft, 
                n_mels=self.n_mel_channels, 
                fmin=self.fmin, 
                fmax=self.fmax)
            mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
            hann_window_cache[key] = torch.hann_window(self.win_length).to(device)

        mel_basis = mel_basis_cache[key]
        hann_window = hann_window_cache[key]

        padding = (self.n_fft - self.hop_length) // 2
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        
        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        # mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        return mel_spec

def make_pad_mask(
    lengths: torch.Tensor, max_len: int = 0, left_pad=False
) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    left_pad:
        A boolean indicating whether to left pad the mask.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    mask = expaned_lengths >= lengths.unsqueeze(-1)

    if left_pad:
        mask = mask.flip(dims=[1])

    return mask

def preprocess_single_inputs(inputs: List[str], max_length=256, device='cuda'):
    """
    Steps to preprocess inputs:
    1. add special geenration tokens after inputs: <|im_start|><image><|im_end|>
    2. add bos token before inputs
    3. tokenize inputs
    4. replace tokens after bos and before <|im_start|> with padding tokens to form unconditional inputs
    5. concatenate conditional inputs and unconditional inputs along batch dimension
    6. create attention masks by masking padding tokens
    7. create noise speech indices
    """

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/gwh/Proj/SmolLM2-360M",
    )
    
    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    tokenizer.add_tokens([DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN, '<|asr|>'], special_tokens=True)
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"

    generation_tokens = f"{DEFAULT_SPEECH_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_SPEECH_END_TOKEN}"

    # inputs = [f"{tokenizer.bos_token}{example}{generation_tokens}" for example in inputs]
    inputs = [f"<|im_start|>{example}{generation_tokens}" for example in inputs]
    
    input_ids = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        # return_tensors="pt",
    )['input_ids']
    
    input_ids = [torch.tensor(i) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    # FIXME: replace pad token after <|im_start|> with <speech>, this is due to tokenizer cannot correctly tokenize <image> after <|im_start|>
    im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)
    im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    for cur_input_ids in input_ids:
        for idx in torch.where(cur_input_ids == im_start_token_id):
            if cur_input_ids[idx + 1] == tokenizer.pad_token_id:
                cur_input_ids[idx + 1] = speech_token_id

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    flags = [[0] for _ in range(len(input_ids))]
    speechs = [[] for _ in range(len(input_ids))]
 
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'flags': flags,
        'speechs': speechs,
    }



def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]



def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] <= end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)



