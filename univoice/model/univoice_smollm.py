import math
from abc import ABC, abstractmethod
from random import random

import torch
from torch import nn
import torch.nn.functional as F

from univoice.constants import *
import torchaudio
import transformers

from .builder import WhisperProjection, WhisperModel
import numpy as np

from univoice.util.utils_smollm import lens_to_mask, mask_from_frac_lengths

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear( hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32
            ) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([
                embedding, torch.zeros_like(embedding[:, :1])
            ], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


def modulate(x, shift, scale, mask):
    return x * (1 + scale.unsqueeze(1) * mask.unsqueeze(2)) + shift.unsqueeze(1) * mask.unsqueeze(2)



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, use_adaln=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(hidden_size, 80, bias=True)
        self.use_adaln = use_adaln
        if self.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            )
        else:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

    def forward(self, x, c_t, flags, mask):
        if self.use_adaln:
            x_indices = [i for i in range(len(flags)) if flags[i] ==0]
            if x_indices==[]:
                x = None
                return x, x_indices
            x = [x[i] for i in x_indices] # x
            mask = [mask[i] for i in x_indices] # mask
            x = torch.stack(x,dim=0)
            mask = torch.stack(mask,dim=0)

            c_t = [c_t[i] for i in range(len(flags)) if flags[i] ==0]
            c_t = torch.stack(c_t,dim=0)
            
            shift, scale = self.adaLN_modulation(c_t).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale, mask)
        # x = self.linear(x.to(dtype=self.linear.weight.dtype)) # deepspeed
        x = self.linear(x)
        return x, x_indices

    def forward_sample(self, x, c_t, flags, mask):
        if self.use_adaln:
            x_indices = [i for i in range(len(flags)) if flags[i] ==0]
            if x_indices==[]:
                x = None
                return x, x_indices
            x = [x[i] for i in x_indices] # x
            mask = [mask[i] for i in x_indices] # mask
            x = torch.stack(x,dim=0)
            mask = torch.stack(mask,dim=0)

            c_t = [c_t[i] for i in range(len(flags)) if flags[i] ==0]
            c_t = torch.stack(c_t,dim=0)
            
            shift, scale = self.adaLN_modulation(c_t).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale, mask)
        x = self.linear(x.to(dtype=self.linear.weight.dtype)) # deepspeed
        return x, x_indices

   

class embed_mel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.x_embedder = nn.Linear(
            in_features =80*2, # 80 for no mask, 80*2 for mask 
            out_features=hidden_size,
            bias=True,)
        self.frac_lengths_mask = (0.7, 1.0)


    def forward(self, x, x_0, target_len, drop_audio_cond, train=True):
        if train: 

            batch, seq_len = x.shape[0], x.shape[1]
            mask = lens_to_mask(target_len, length=seq_len) 
            # get a random span to mask out for training conditionally
            frac_lengths = torch.zeros((batch,)).float().uniform_(*self.frac_lengths_mask)
            rand_span_mask = mask_from_frac_lengths(target_len, frac_lengths)
            rand_span_mask &= mask

            # only predict what is within the random mask span for infilling
            cond = torch.where(rand_span_mask[..., None].to(x.device), torch.zeros_like(x_0).to(x.device), x_0)


            if drop_audio_cond:  # cfg for cond audio
                cond = torch.zeros_like(cond).to(x.device)
            

            x_embed = self.x_embedder(torch.cat((x, cond), dim=-1))

            return x_embed, rand_span_mask
        
        else:
            batch, seq_len = x.shape[0], x.shape[1]

            lens = torch.full((batch,), seq_len, device=x.device, dtype=torch.long)
            cond_mask = lens_to_mask(lens)

            x_embed = self.x_embedder(torch.cat((x, x_0), dim=-1).to(self.x_embedder.weight.dtype))

            return x_embed, cond_mask


class UniVoiceMetaModel:
    def __init__(self, config):
        super(UniVoiceMetaModel, self).__init__(config)

        self.audio_encoder = WhisperModel.from_pretrained(config.whisper_path).get_encoder()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.audio_embedder = WhisperProjection(input_embedding_size=1280, output_embedding_size=config.hidden_size)


        self.adaLN_module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        )
        
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.x_embedder = embed_mel(hidden_size=config.hidden_size)
        self.t_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True))
        self.final_layer = FinalLayer(config.hidden_size, use_adaln=True)
        
    def initialize_weights(self):
        """
        Call this function to initialize the additional modules for DiT generation after loading pretrained weights.
        """
        for m in self.final_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.t_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.x_embedder.x_embedder.weight, 0)
        nn.init.constant_(self.x_embedder.x_embedder.bias, 0)

        if hasattr(self, 't_projector'):
            for m in self.t_projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'adaLN_module'):
            for m in self.adaLN_module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
    

class UniVoiceMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_audio_encoder(self):
        return self.get_model().audio_encoder

    def get_audio_embedder(self):
        return self.get_model().audio_embedder

    def get_t_embedder(self):
        return self.get_model().t_embedder

        
    def get_t_projector(self):
        return self.get_model().t_projector
        
    def get_x_embedder(self):
        return self.get_model().x_embedder
    

    def get_final_layer(self):
        return self.get_model().final_layer
    


    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, speechs, t, x_t,x_0, flags, target_len, **kwargs
    ):
        # print('input_ids:',input_ids.shape) # (2, T) for tts, (1,T) for asr
        if input_ids.shape[1] == 1:
         
            model_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                'inputs_embeds': None,
                "labels": labels,
                'speech_token_spans': None,
                'c_embeds': None,
                't_embeds': t,
                'c_embeds_mask': None,
                "rand_span_mask": None
            }
            return model_inputs

        if speechs is None:
            speechs = []
        


        speech_inputs = []
        x_t_inputs = []
        x_0_inputs = []
        # generate
        for i in range(flags.shape[0]):
            if flags[i] == 0:
                x_t_inputs.append(x_t[i])
                x_0_inputs.append(x_0[i])
        # understanding
        for i in range(flags.shape[0]):
            if flags[i] == 1:
                speech_inputs.append(speechs[i])


        

        drop_audio_cond = random() < 0.3  # p_drop in voicebox paper
        if random() < 0.2:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        new_input_ids = []
        if drop_text: 
            for i in range(input_ids.shape[0]):
                if flags[i] == 0:
                    seq = torch.tensor(self.tokenizer.convert_tokens_to_ids(["<|im_start|>", DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_END_TOKEN]))
                else:
                    seq = input_ids[i]
                new_input_ids.append(seq)


            
            input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids,padding_value=0, batch_first=True).to(attention_mask.device)
            labels = torch.ones_like(input_ids).to(attention_mask.device)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)


        x_t_embeds = None
        rand_span_mask = None
        if len(x_t_inputs) != 0:
            target_len_0 = torch.tensor([target_len[i] for i in range(target_len.shape[0]) if flags[i]==0],dtype=torch.int32)

            x_t_inputs = torch.stack(x_t_inputs)
            x_0_inputs = torch.stack(x_0_inputs)
            x_t_embeds, rand_span_mask = self.get_x_embedder()(x_t_inputs,x_0_inputs,target_len_0,drop_audio_cond) # TODO: F5 mask style in get_x_embedder()

        # asr audio encoder
        if len(speech_inputs) > 0:

            speech_features = self.get_audio_encoder()(torch.stack(speech_inputs).transpose(1,2)).last_hidden_state

            speech_features = self.get_audio_embedder()(speech_features)


        else:
            speech_features = []


        speech_embeds = []
        speech_gen_idx = 0
        speech_und_idx = 0


        for i in range(flags.shape[0]):
            # speech_size = None
            if flags[i] == 0:
                speech_embeds.append(x_t_embeds[speech_gen_idx])
                speech_gen_idx += 1
            elif flags[i] == 1:
                speech_embeds.append(speech_features[speech_und_idx])
                speech_und_idx += 1



        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        
        speech_token_spans = []
        cur_speech_idx = 0
        
        t_embeds = self.get_t_embedder()(t)  # (batch_size, hidden_size)

        if self.config.decoder_t_embed == 'add_before_speech_tokens':
            t_tokens = self.get_t_projector()(t_embeds)  # (batch_size, hidden_size)

        for batch_idx, cur_input_ids in enumerate(input_ids):

            num_speechs = (cur_input_ids == self.config.speech_token_index).sum()

            # print('num_speechs:',num_speechs)
            if num_speechs == 0:
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                speech_token_spans.append([])
                continue
            
            speech_token_indices = [-1] + torch.where(cur_input_ids == self.config.speech_token_index)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(speech_token_indices) - 1):
                
                cur_input_ids_noim.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_speech_token_spans = (0,0)

            for i in range(num_speechs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_speechs:
                    if self.config.decoder_t_embed == 'add_before_speech_tokens':
                        cur_new_input_embeds.append(t_tokens[batch_idx:batch_idx+1]) # add t condition
                        cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_speech_features = speech_embeds[cur_speech_idx]
                    cur_speech_idx += 1 
                    speech_token_start_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_input_embeds.append(cur_speech_features)
                    speech_token_end_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    
                    if flags[batch_idx] == 0:
                        cur_speech_token_spans = (speech_token_start_idx, speech_token_end_idx)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            speech_token_spans.append(cur_speech_token_spans)

        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left": # for qwen padding_side
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        #===================================
        #  generate condition embedding mask
        #===================================
        c_embeds=None
        c_embeds_mask = torch.zeros(new_input_embeds.shape[:2]).to(t_embeds.device)  # (batch_size, seq_len)

        for i, span in enumerate(speech_token_spans):
            if span[1]!=0:
                c_embeds_mask[i, span[0]:span[1]] = 1
        
        if self.config.decoder_t_embed == 'adaln_before_decoder':
            c_embeds = self.get_model().adaLN_module(t_embeds).to(t_embeds.device)
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        model_inputs = {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_input_embeds,
            'labels': new_labels,
            'speech_token_spans': speech_token_spans,
            't_embeds': t_embeds,
            'c_embeds_mask': c_embeds_mask,
            'c_embeds': c_embeds,
            'rand_span_mask': rand_span_mask,
        }


        return model_inputs



    def prepare_inputs_labels_for_multimodal_for_inference(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, speechs, t, cond, step_cond, flags, target_len, drop_audio_cond, drop_text, **kwargs
    ):
        # print('input_ids:',input_ids.shape) # (2, T) for tts, (1,T) for asr
        if input_ids.shape[1] == 1:
         
            model_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                'inputs_embeds': None,
                "labels": labels,
                'speech_token_spans': None,
                'c_embeds': None,
                't_embeds': t,
                'c_embeds_mask': None,
            }
            return model_inputs

        if speechs is None:
            speechs = []
        
        speech_inputs = []
        x_t_inputs = []
        # generate
        for i in range(flags.shape[0]):
            if flags[i] == 0:
                x_t_inputs.append(cond[i])
        # understanding
        for i in range(flags.shape[0]):
            if flags[i] == 1:
                speech_inputs.append(speechs[i])
        
        new_input_ids = []
        if drop_text: # NOTE! now is error, only for tts task drop text or drop_audio
            for i in range(input_ids.shape[0]):
                if flags[i] == 0:
                    seq = torch.tensor(self.tokenizer.convert_tokens_to_ids(["<|im_start|>", DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_END_TOKEN]))
                else:
                    seq = input_ids[i]
                new_input_ids.append(seq)
            
            input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids,padding_value=0, batch_first=True).to(attention_mask.device)
            labels = torch.ones_like(input_ids).to(attention_mask.device)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)


        x_t_embeds = None
        if len(x_t_inputs) != 0:
            target_len_0 = torch.tensor([target_len[i] for i in range(target_len.shape[0]) if flags[i]==0],dtype=torch.int32)
            x_t_inputs = torch.stack(x_t_inputs)
            x_t_embeds, rand_span_mask = self.get_x_embedder()(x_t_inputs, step_cond, target_len_0, drop_audio_cond, train=False) # TODO: F5 mask style in get_x_embedder()

        # asr audio encoder
        if len(speech_inputs) > 0:
            # whisper online train
            speech_features = self.get_audio_encoder()(torch.stack(speech_inputs).transpose(1,2)).last_hidden_state
            speech_features = self.get_audio_embedder()(speech_features)

        else:
            speech_features = []


        speech_embeds = []
        speech_gen_idx = 0
        speech_und_idx = 0


        for i in range(flags.shape[0]):
            if flags[i] == 0:
                speech_embeds.append(x_t_embeds[speech_gen_idx])
                speech_gen_idx += 1
            elif flags[i] == 1:
                speech_embeds.append(speech_features[speech_und_idx])
                speech_und_idx += 1


        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        
        speech_token_spans = []
        cur_speech_idx = 0
        
        t_embeds = self.get_t_embedder()(t)  # (batch_size, hidden_size)

        if self.config.decoder_t_embed == 'add_before_speech_tokens':
            t_tokens = self.get_t_projector()(t_embeds)  # (batch_size, hidden_size)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speechs = (cur_input_ids == self.config.speech_token_index).sum()
            if num_speechs == 0:
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                speech_token_spans.append([])
                continue
            
            speech_token_indices = [-1] + torch.where(cur_input_ids == self.config.speech_token_index)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_speech_token_spans = (0,0)

            for i in range(num_speechs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_speechs:
                    if self.config.decoder_t_embed == 'add_before_speech_tokens':
                        cur_new_input_embeds.append(t_tokens[batch_idx:batch_idx+1]) # add t condition
                        cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_speech_features = speech_embeds[cur_speech_idx]
                    cur_speech_idx += 1 
                    speech_token_start_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_input_embeds.append(cur_speech_features)
                    speech_token_end_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    
                    if flags[batch_idx] == 0:
                        cur_speech_token_spans = (speech_token_start_idx, speech_token_end_idx)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            speech_token_spans.append(cur_speech_token_spans)

        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        #===================================
        #  generate condition embedding mask
        #===================================
        c_embeds_mask = torch.zeros(new_input_embeds.shape[:2]).to(t_embeds.device)  # (batch_size, seq_len)
        c_embeds = None
        for i, span in enumerate(speech_token_spans):
            if span[1]!=0:
                c_embeds_mask[i, span[0]:span[1]] = 1
        
        if self.config.decoder_t_embed == 'adaln_before_decoder':
            c_embeds = self.get_model().adaLN_module(t_embeds).to(t_embeds.device)
        
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        model_inputs = {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_input_embeds,
            'labels': new_labels,
            'speech_token_spans': speech_token_spans,
            't_embeds': t_embeds,
            'c_embeds_mask': c_embeds_mask,
            'c_embeds': c_embeds,
            'rand_span_mask': rand_span_mask,
        }

        return model_inputs

