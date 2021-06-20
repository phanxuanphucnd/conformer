# -*- coding: utf-8 -*-

import torch

from torch import Tensor
from typing import Tuple, Optional

from arizona_asr.models.model import TransducerModel

class Conformer(TransducerModel):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.
    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        decoder_dim (int, optional): Dimension of conformer decoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_decoder_layers (int, optional): Number of decoder layers
        decoder_rnn_type (str, optional): type of RNN cell
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        decoder_dropout_p (float, optional): Probability of conformer decoder dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
        device (torch.device): torch device (cuda or cpu)
        decoder (str): If decoder is None, train with CTC decoding
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """

    