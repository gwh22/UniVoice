# modified form https://github.com/MonoFormer/MonoFormer
from typing import Optional, List, Union, Tuple
import warnings

import numpy as np
import torch
from torch import nn
from transformers import LlamaModel, Qwen2Model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import StaticCache, DynamicCache, Cache
import transformers
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

def modulate(x, shift, scale, mask):
    return x * (1 + scale.unsqueeze(1) * mask.unsqueeze(2)) + shift.unsqueeze(1) * mask.unsqueeze(2)


def modulate_adaln(x, emb, mask):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

    x = x * (1 + scale_msa[:, None] * mask.unsqueeze(2)) + shift_msa[:, None] * mask.unsqueeze(2)
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

# LlamaModel
class SmolLMModel(LlamaModel): 
    def __init__(self, config):
        super().__init__(config)
        if self.config.use_pos_embed:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.config.hidden_size, np.arange(4096, dtype=np.float32))
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.register_buffer('pos_embed', pos_embed, persistent=False)
    
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        c_embeds: Optional[torch.FloatTensor] = None,
        c_embeds_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.pos_embed = self.pos_embed.to(attention_mask.device)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4.39.0
        past_seen_tokens = 0
        if use_cache:  
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None or cache_position.shape  == torch.Size([0]):
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
    
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
       
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position) # for  transformers==4.39.0


        if  c_embeds_mask is not None:
            mask_length = c_embeds_mask.shape[-1]
            speech_attn_mask = ~torch.logical_and(c_embeds_mask[:, None, None, :].repeat(1, 1, mask_length, 1).eq(1.0), c_embeds_mask[:, None, None, :].repeat(1, 1, mask_length, 1).transpose(2, 3).eq(1.0))
            causal_mask = causal_mask[..., :mask_length, :mask_length] * speech_attn_mask
        
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if c_embeds_mask is not None:
                if self.config.use_bi_attn_speech_tokens:
                    if self.config.use_pos_embed and not self.config.add_pos_embed_each_layer and layer_idx == 0:
                        temp_c_embeds_mask = torch.cat([c_embeds_mask, torch.zeros(c_embeds_mask.shape[0], 1, device=c_embeds_mask.device)], dim=1)
                        indices = torch.where((temp_c_embeds_mask[:, 1:] - temp_c_embeds_mask[:, :-1]) != 0)

                        for j in range(0, len(indices[0]), 2):
                            batch_idx = indices[0][j]
                            start_idx = indices[1][j] + 1
                            end_idx = min(indices[1][j+1] + 1, hidden_states.shape[1]-1)
                            hidden_states[batch_idx, start_idx:end_idx] = hidden_states[batch_idx, start_idx:end_idx] + self.pos_embed[0, :end_idx-start_idx]

      
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,

                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,

                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    