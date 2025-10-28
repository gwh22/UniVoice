# modified form https://github.com/MonoFormer/MonoFormer
from transformers import LlamaForCausalLM
import math
from torch import nn
import functools
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import warnings
import random
import os
import numpy as np


import torch
import transformers
import torchaudio
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from univoice.constants import *
from univoice.model.language_model.modeling_smollm import SmolLMModel

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from ..univoice_smollm import UniVoiceMetaForCausalLM, UniVoiceMetaModel
from univoice.util.utils_smollm import preprocess_single_inputs
from univoice.util.utils_smollm import lens_to_mask

@dataclass
class UniVoiceCausalLMOutputWithPast(CausalLMOutputWithPast):
    fm_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    x_out: Optional[torch.FloatTensor] = None


class UniVoiceSmolLMConfig(LlamaConfig):
    model_type = "univoice-smollm"
    use_pos_embed = True
    use_bi_attn_speech_tokens = True
    speech_token_index = None
    decoder_t_embed = 'add_before_speech_tokens'
    add_pos_embed_each_layer = False
    sample_N = 32
    whisper_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/gwh/Proj/whisper-large-v3-turbo"



class UniVoiceSmolLMModel(UniVoiceMetaModel, SmolLMModel):
    config_class = UniVoiceSmolLMConfig

    def __init__(self, config: LlamaConfig):
        super(UniVoiceSmolLMModel, self).__init__(config)
        

class UniVoiceForCausalLM(LlamaForCausalLM, UniVoiceMetaForCausalLM):
    config_class = UniVoiceSmolLMConfig
    def __init__(self, config):
        super(UniVoiceForCausalLM, self).__init__(config)
        self.model = UniVoiceSmolLMModel(config)
        # flow matching
        self.sample_N = config.sample_N
        self.T = 1
        self.eps = 1e-3
        self.config = config

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/gwh/Proj/SmolLM2-360M",
        )
        self.tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
        self.tokenizer.add_tokens([DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN, '<|asr|>'], special_tokens=True)
        self.tokenizer.bos_token = "<|im_start|>"
        self.tokenizer.eos_token = "<|im_end|>"
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.model.layers)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        text: Optional[str] = None,
        speechs: Optional[List[torch.FloatTensor]] = None,
        mel_spec: Optional[List[torch.FloatTensor]] = None,
        t: Optional[torch.FloatTensor] = None,
        x_t: Optional[torch.FloatTensor] = None,
        target_len: Optional[torch.Tensor] = None,
        flags: Optional[List[torch.LongTensor]] = None,
        # args for generate
        speech_token_spans: Optional[List[Tuple[int, int]]] = None,
        speech_sizes: Optional[List[Tuple[int, int]]] = None,
        t_embeds: Optional[torch.FloatTensor] = None,
        c_embeds: Optional[torch.FloatTensor] = None,
        c_embeds_mask: Optional[torch.LongTensor] = None,
        rand_span_mask: Optional[torch.LongTensor] = None,
        training: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # preprocess
        x_0 = None
        if training:
            x_0 = mel_spec
            t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - self.eps) + self.eps
            t_expand = t.view(-1, 1, 1).repeat(
                1, x_0.shape[1], x_0.shape[2]
            )
            noise_tensor = torch.randn_like(mel_spec)
            target = x_0 - noise_tensor
            perturbed_data = t_expand * x_0 + (1 - t_expand) * noise_tensor

            x_t = perturbed_data
            t = t * 1000.
        # for inference 
        if inputs_embeds is None:
            model_inputs = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, speechs, t, x_t, x_0, flags, target_len)
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, speech_token_spans, t_embeds, c_embeds_mask,c_embeds, rand_span_mask = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['speech_token_spans'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds_mask'],
                model_inputs['c_embeds'],
                model_inputs['rand_span_mask']
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            c_embeds=c_embeds,
            c_embeds_mask=c_embeds_mask,
        )

        hidden_states = outputs['last_hidden_state'] 

        '''
        Compute outputs for generation losses
        '''
        x_out = []
        B = len(hidden_states)
        x_indices = []

        if c_embeds_mask is not None:
            x, x_indices = self.get_final_layer()(hidden_states, t_embeds, flags, c_embeds_mask)
            speech_token_spans = [speech_token_spans[i] for i in range(len(speech_token_spans)) if speech_token_spans[i] != (0,0)]
            if x == None:
                x_out=[]
            else:
                for i in range(x.shape[0]):
                    span = speech_token_spans[i]
                    if span:
                        speech_tokens = x[i, span[0]:span[1], :]
                        x_out.append(speech_tokens)

        x_indices = torch.tensor(x_indices)

        ''' 
        Compute the language modeling loss
        '''
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = 0.
        lm_loss = 0.
        if labels is not None:
            # only compute loss for samples not used for diffusion generation
            lm_indices = []
            lm_indices = torch.tensor([i for i in range(len(hidden_states)) if i not in x_indices], dtype=torch.long)
            if len(lm_indices) == 0:
                lm_loss = torch.tensor(0., device=logits.device)
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[lm_indices, :-1, :].contiguous()
                shift_labels = labels[lm_indices, 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)

        if x_out==[]:

            loss = lm_loss*0.005

        fm_loss = 0.
        if x_out!=[]:
            x_out = torch.stack(x_out,dim=0)
            target = torch.stack([target[i] for i in x_indices],dim=0)

            fm_loss = F.mse_loss(x_out, target, reduction="none")  
            fm_loss = fm_loss[rand_span_mask].mean() 
            loss = fm_loss+lm_loss*0.005
        
        return UniVoiceCausalLMOutputWithPast(
            loss=loss,
            fm_loss=fm_loss,
            lm_loss=lm_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def forward_sample(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        text: Optional[str] = None,
        speechs: Optional[List[torch.FloatTensor]] = None,
        mel_spec: Optional[List[torch.FloatTensor]] = None,
        t: Optional[torch.FloatTensor] = None,
        x_t: Optional[torch.FloatTensor] = None,
        step_cond: Optional[torch.FloatTensor] = None,
        target_len: Optional[torch.Tensor] = None,
        flags: Optional[List[torch.LongTensor]] = None,
        # args for generate
        speech_token_spans: Optional[List[Tuple[int, int]]] = None,
        t_embeds: Optional[torch.FloatTensor] = None,
        c_embeds: Optional[torch.FloatTensor] = None,
        c_embeds_mask: Optional[torch.LongTensor] = None,
        drop_audio_cond: Optional[bool] = None,
        drop_text: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # preprocess
        # for inference 
        if inputs_embeds is None:
            model_inputs = self.prepare_inputs_labels_for_multimodal_for_inference(input_ids, position_ids, attention_mask, past_key_values, labels, speechs, t, x_t, step_cond, flags, target_len, drop_audio_cond, drop_text)
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, speech_token_spans, t_embeds, c_embeds, c_embeds_mask = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['speech_token_spans'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds'],
                model_inputs['c_embeds_mask'],
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            c_embeds=c_embeds,
            c_embeds_mask=c_embeds_mask,
        )

        hidden_states = outputs['last_hidden_state'] 

        '''
        Compute outputs for generation losses
        '''
        x_out = []
        B = len(hidden_states)
        x_indices = []

        if c_embeds_mask is not None:
            x, x_indices = self.get_final_layer().forward_sample(hidden_states, t_embeds, flags, c_embeds_mask) 

            speech_token_spans = [speech_token_spans[i] for i in range(len(speech_token_spans)) if speech_token_spans[i] != (0,0)]
            if x == None:
                x_out=[]
            else:
                for i in range(x.shape[0]):
                    span = speech_token_spans[i]
                    if span:
                        speech_tokens = x[i, span[0]:span[1], :]
                        x_out.append(speech_tokens)

        x_indices = torch.tensor(x_indices)

        ''' 
        Compute the language modeling loss
        '''
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        
        
        return UniVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            x_out=x_out,
        )
       
    @torch.no_grad()
    def sample(
        self, input_ids, attention_mask, labels, mel_spec, speechs, flags, target_len, text, cfg_scale
    ):
        for i in range(1):
            z1 = mel_spec[i]
            z2 = z1[:(z1.shape[0]//2),:].unsqueeze(0)

            mel_gt = mel_spec[i]
            text = text[i]
            y = [text]

            inputs_dict = preprocess_single_inputs(y)
            target_len = torch.tensor([target_len[i]],dtype=torch.int32)


            batch, seq_len = z2.shape[0], z2.shape[1]

            lens = torch.full((batch,), seq_len, device=z2.device, dtype=torch.long)
            cond_mask = lens_to_mask(lens)

            max_duration = max(target_len)
        
            cond = F.pad(z2, (0, 0, 0, max_duration - seq_len), value=0.0)

            cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
            cond_mask = cond_mask.unsqueeze(-1)

            step_cond = torch.where(
                cond_mask, cond, torch.zeros_like(cond)
            )  # allow direct control (cut cond audio) with lens passed in

            if batch > 1:
                mask = lens_to_mask(target_len)
            else:  # save memory and speed up, as single inference need no mask currently
                mask = None
            
            y0 = []
            for dur in target_len:
                y0.append(torch.randn(dur, 80, device=z2.device, dtype=step_cond.dtype))
            y0 = torch.nn.utils.rnn.pad_sequence(y0, padding_value=0, batch_first=True)

            with torch.no_grad():
                mel_out,nfe = self.euler_sample(
                    input_ids=inputs_dict['input_ids'],
                    attention_mask=inputs_dict['attention_mask'],
                    labels=inputs_dict['input_ids'],
                    mel_spec=y0,
                    step_cond=step_cond,
                    speechs=speechs,
                    flags=torch.tensor(inputs_dict['flags'],dtype=torch.int32),
                    target_len=target_len,
                    guidance_scale=cfg_scale)
            mel_out = mel_out[:target_len[i],:]
            mel_out = torch.where(cond_mask,cond,mel_out) # for f5-style
        return mel_out, mel_gt




    @torch.no_grad()
    def sample_ref_tar(
        self, input_ids, attention_mask, labels, mel_spec_ref, speechs, flags, target_len, text, cfg_scale
    ):
        for i in range(1):
            z1 = mel_spec_ref
            text = text[i]
            y = [text]
            inputs_dict = preprocess_single_inputs(y)
            target_len = torch.tensor([target_len[i]],dtype=torch.int32)


            batch, seq_len = z1.shape[0], z1.shape[1]
            # print('x:',x.shape)
            lens = torch.full((batch,), seq_len, device=z1.device, dtype=torch.long)
            cond_mask = lens_to_mask(lens)
            max_duration = max(target_len)
        
            cond = F.pad(z1, (0, 0, 0, max_duration - seq_len), value=0.0)

            cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
            cond_mask = cond_mask.unsqueeze(-1)
            step_cond = torch.where(
                cond_mask, cond, torch.zeros_like(cond)
            )  # allow direct control (cut cond audio) with lens passed in

            if batch > 1:
                mask = lens_to_mask(target_len)
            else:  # save memory and speed up, as single inference need no mask currently
                mask = None
            
            y0 = []
            for dur in target_len:
                y0.append(torch.randn(dur, 80, device=z1.device, dtype=step_cond.dtype))
            y0 = torch.nn.utils.rnn.pad_sequence(y0, padding_value=0, batch_first=True)

            # cfg_scale = 4
            with torch.no_grad():
                mel_out,nfe = self.euler_sample(
                    input_ids=inputs_dict['input_ids'],
                    attention_mask=inputs_dict['attention_mask'],
                    labels=inputs_dict['input_ids'],
                    mel_spec=y0,
                    step_cond=step_cond,
                    speechs=speechs,
                    flags=torch.tensor(inputs_dict['flags'],dtype=torch.int32),
                    target_len=target_len,
                    guidance_scale=cfg_scale)
            mel_out = mel_out[:target_len[i],:]
            mel_out = torch.where(cond_mask,cond,mel_out)
        return mel_out



    @torch.no_grad()
    def euler_sample(
        self, 
        input_ids,
        attention_mask,
        labels,
        speechs,
        mel_spec,
        step_cond,
        flags,
        target_len,
        guidance_scale
    ):
        device = self.model.device
        # uniform
        dt = 1.0 / self.sample_N
        eps = 1e-3
        t = torch.linspace(0, 1, self.sample_N + 1, device=self.device, dtype=step_cond.dtype)
        sway_sampling_coef = -1.0
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        t_copy = t
        for ind,i in enumerate(t[:-1]):
            # print(ind, i)
            num_t = i * (self.T - eps) + eps
            t = torch.ones(len(mel_spec), device=device) * num_t

            dt = t_copy[ind+1]-t_copy[ind]

            cond_pred = self.forward_sample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                speechs=speechs,
                flags=flags,
                t=t * 1000,
                x_t=mel_spec,
                step_cond=step_cond,
                target_len=target_len,
                drop_audio_cond=False, 
                drop_text=False
            )['x_out'][0]
            uncond_pred = self.forward_sample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                speechs=speechs,
                flags=flags,
                t=t * 1000,
                x_t=mel_spec,
                step_cond=step_cond,
                target_len=target_len,
                drop_audio_cond=True, 
                drop_text=True
            )['x_out'][0]

            # perform guidance
            pred = cond_pred + (cond_pred - uncond_pred) * guidance_scale

            # cfg-renorm
            ori_pos_norm = torch.linalg.vector_norm(cond_pred
                    , dim=tuple(range(0, len(cond_pred.shape))), keepdim=True
            )
            max_new_norm = ori_pos_norm * float(1.0)
            new_pos_norm = torch.linalg.vector_norm(
                pred, dim=tuple(range(0, len(pred.shape))), keepdim=True
            )
            if new_pos_norm >= max_new_norm:
                pred = pred * (max_new_norm / new_pos_norm)

            pred_sigma = pred
            mel_spec = (
                mel_spec.detach().clone()
                + pred_sigma * dt
            )
        nfe = self.sample_N
        return mel_spec, nfe

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        t = kwargs.pop("t", None)
        speechs = kwargs.pop("speechs", None)
        x_t = kwargs.pop("x_t", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        flags = kwargs.pop("flags", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        is_multimodal = True
        
        if is_multimodal:
            model_inputs = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                speechs,
                t,
                x_t,
                x_t,
                flags,
                None, 
            )
            _, position_ids, attention_mask, _, inputs_embeds, _, speech_token_spans, t_embeds, c_embeds, c_embeds_mask,rand_span_mask  = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['speech_token_spans'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds'],
                model_inputs['c_embeds_mask'],
                model_inputs['rand_span_mask'],
            )
            kwargs.update({
                'speech_token_spans': speech_token_spans,
                't_embeds': t_embeds,
                'c_embeds': c_embeds,
                'c_embeds_mask': c_embeds_mask,
                'flags': flags,
                'speechs': speechs
            })
        else:
            inputs_embeds = self.model.embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            do_sample=True,
            max_new_tokens=200,
            use_cache=True,
            training=False, ###
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs.update(kwargs)
        return inputs




AutoConfig.register("univoice-smollm", UniVoiceSmolLMConfig)
AutoModelForCausalLM.register(UniVoiceSmolLMConfig, UniVoiceForCausalLM)
