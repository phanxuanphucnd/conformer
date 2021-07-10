# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class ModelConfig:
    architecture: str='???'
    teacher_forcing_ratio: float=1.0
    teacher_forcing_step: float=0.01
    min_teacher_forcing_ratio: float=0.9
    dropout: float=0.3
    bidirectional: bool=False
    joint_ctc_attention: bool=False
    max_len: int=400
