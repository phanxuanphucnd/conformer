# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class RNNTransducerConfig:
    architecture: str = "rnnt"
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1
    encoder_hidden_state_dim: int = 320
    decoder_hidden_state_dim: int = 512
    output_dim: int = 512
    rnn_type: str = "lstm"
    bidirectional: bool = True
    encoder_dropout_p: float = 0.2
    decoder_dropout_p: float = 0.2