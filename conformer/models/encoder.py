# -*- coding: utf-8 -*-

import torch.nn as nn

from torch import Tensor
from typing import Tuple

from arizona_asr.models.modules import Transpose, Linear
from arizona_asr.models.convolutional import VGGExtractor, DeepSpeech2Extractor

class EncoderInterface(nn.Module):
    """Base interface of Encoder. """
    def __init__(self):
        super(EncoderInterface, self).__init__()

    def count_parameters(self) -> int:
        """Count parameters of encoder. """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """Update dropout probability of encoder. """
        for _, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p
    
    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        Forward propagate for encoder training.
        
        Args:
            inputs: A input sequence passed to encoder. Typically for inputs this will be a 
                    padded `FloatTensor`of size ``(batch, seq_length, dimension)``.
            input_lengths: The length of input tensor. ``(batch)``

        """
        raise NotImplementedError

class BaseEncoder(EncoderInterface):
    """ASR Encoder super class for ViSpeech model implementation. """
    supported_extractors = {
        'vgg': VGGExtractor,
        'ds2': DeepSpeech2Extractor,
    }

    def __init__(
        self,
        input_dim: int,
        extractor: str='vgg',
        d_model: int=None,
        num_classes: int=None,
        dropout_p: float=None,
        activation: str='hardtanh',
        joint_ctc_attention: bool=False
    ) -> None:
        super(BaseEncoder, self).__init__()
        
        if joint_ctc_attention:
            assert num_classes, "If `joint_ctc_attention` True, `num_classes` should be not None"
            assert dropout_p, "If `joint_ctc_attention` True, `dropout_p` should be not None"
            assert d_model, "If `joint_ctc_attention` True, `d_model` should be not None"

        if extractor:
            extractor = self.supported_extractors[extractor.lower()]
            self.conv = extractor(input_dim=input_dim, activation=activation)

        self.conv_output_dim = self.conv.get_output_dim()
        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention

        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(d_model),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(d_model, num_classes, bias=False)
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor, Tensor):
            * encoder_outputs: A output sequence of encoder. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * encoder_output_lengths: The length of encoder outputs. ``(batch)``
            * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        """
        raise NotImplementedError

class TransducerEncoder(EncoderInterface):
    """ASR Transducer Encoder Super class for ViSpeech model implementation. """
    def __init__(self):
        super(TransducerEncoder, self).__init__()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        raise NotImplementedError