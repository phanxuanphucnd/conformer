# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple

from arizona_asr.models.encoder import TransducerEncoder
from arizona_asr.models.convolutional import Conv2dSubsampling
from arizona_asr.models.modules import ResidualConnectionModule, Linear