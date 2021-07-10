# -*- coding: utf-8 -*-

from dataclasses import dataclass
from arizona_asr.models import ModelConfig
from arizona_asr.models.conformer.model import Conformer


@dataclass
class ConformerConfig(ModelConfig):
    architecture: str = "conformer"
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    decoder_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    half_step_residual: bool = True
    num_decoder_layers: int = 1
    decoder_rnn_type: str = "lstm"
    decoder: str = "None"


@dataclass
class ConformerLargeConfig(ConformerConfig):
    encoder_dim: int = 512
    decoder_dim: int = 640
    num_encoder_layers: int = 17
    num_attention_heads: int = 8


@dataclass
class ConformerMediumConfig(ConformerConfig):
    encoder_dim: int = 256
    decoder_dim: int = 640
    num_encoder_layers: int = 16
    num_attention_heads: int = 4


@dataclass
class ConformerSmallConfig(ConformerConfig):
    encoder_dim: int = 144
    decoder_dim: int = 320
    num_encoder_layers: int = 16
    num_attention_heads: int = 4