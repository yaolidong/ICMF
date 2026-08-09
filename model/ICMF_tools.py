

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import math

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .Tool_model import GAT




class MultiModalEncoder(nn.Module):
    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 attr_input_dim=1000,
                 ):
        super(MultiModalEncoder, self).__init__()

        attr_dim = args.attr_dim
        img_dim = args.img_dim
        n_units = [int(x) for x in args.hidden_units.strip().split(",")]
        n_heads = [int(x) for x in args.heads.strip().split(",")]

        self.entity_emb = nn.Embedding(ent_num, n_units[0])
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(ent_num))
        self.entity_emb.requires_grad = True

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)


        self.name_fc = nn.Linear(300, 300)
        self.char_fc = nn.Linear(char_feature_dim, 300)

        self.cross_graph_model = GAT(
            n_units=n_units,
            n_heads=n_heads,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
        )








class CrossModalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if key_mask is not None:
            key_mask = torch.as_tensor(key_mask, dtype=torch.bool, device=attention_scores.device)
            mask = (~key_mask).unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask, torch.finfo(attention_scores.dtype).min)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class CrossModalAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = CrossModalSelfAttention(config)
        self.output = _SelfOutput(config)

    def forward(self, hidden_states, key_mask=None):
        self_output = self.self(hidden_states, key_mask=key_mask)
        return self.output(self_output, hidden_states)


class CrossModalLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.cml_mode = str(getattr(config, "cml_mode", "full"))
        self.attention = CrossModalAttention(config)
        if self.config.use_intermediate:
            self.intermediate = _Intermediate(config)
        self.output = _FeedForwardOutput(config)

    def forward(self, hidden_states, key_mask=None):


        if self.cml_mode == "layernorm_only":
            return self.attention.output.LayerNorm(hidden_states)

        attention_output = self.attention(hidden_states, key_mask=key_mask)
        if not self.config.use_intermediate:
            return attention_output

        return apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output





class _SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class _Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class _FeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

        modal_num = 3

        gph_mode = str(config.gph_interaction_mode)
        if gph_mode != 'none':
            modal_num += 1
        self.conv = ConvModule(in_channels=modal_num)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, bias):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        return self.conv(inputs)


class ConvModule(nn.Module):
    def __init__(self, in_channels, kernel_size=31, expansion_factor=2):
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2

        self.sequential = nn.Sequential(
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
        )

    def forward(self, inputs):
        return self.sequential(inputs)
