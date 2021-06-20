import logging
import pandas as pd
import torch.nn as nn

from queue import Queue

from arizona_asr.models.transformer.decoder import TransformerDecoder
from arizona_asr.models.model import EncoderDecoderModel, EncoderModel
from arizona_asr.models.beam_search import BeamCTCDecoder, BeamDecoderRNN, BeamTransformerDecoder
from arizona_asr.metrics import (
    CharacterErrorRate,
    WordErrorRate
)

logger = logging.getLogger(__name__)

class GreedySearch(object):
    """ Provides greedy search and save result to csv format """
    def __init__(self, vocab, metric: str = 'char'):
        self.target_list = list()
        self.predict_list = list()
        self.vocab = vocab

        if metric == 'char':
            self.metric = CharacterErrorRate(vocab)
        elif metric == 'word':
            self.metric = WordErrorRate(vocab)
        else:
            raise ValueError("Unsupported metric : {0}".format(metric))

    def search(self, model: nn.Module, queue: Queue, device: str, print_every: int) -> float:
        cer = 0
        total_sent_num = 0
        timestep = 0

        if isinstance(model, nn.DataParallel):
            model = model.module

        model.eval()
        model.to(device)

        while True:
            inputs, targets, input_lengths, target_lengths = queue.get()

            if inputs.shape[0] == 0:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            y_hats = model.recognize(inputs, input_lengths)

            for idx in range(targets.size(0)):
                self.target_list.append(self.vocab.label_to_string(targets[idx]))
                self.predict_list.append(self.vocab.label_to_string(y_hats[idx].cpu().detach().numpy()))

            cer = self.metric(targets[:, 1:], y_hats)
            total_sent_num += targets.size(0)

            if timestep % print_every == 0:
                logger.info('cer: {:.2f}'.format(cer))

            timestep += 1

        return cer

    def save_result(self, save_path: str) -> None:
        results = {
            'targets': self.target_list,
            'predictions': self.predict_list
        }
        results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, encoding='cp949')


class BeamSearch(GreedySearch):
    """ Provides beam search decoding. """
    def __init__(self, vocab, k: int, batch_size: int):
        super(BeamSearch, self).__init__(vocab)
        self.k = k
        self.batch_size = batch_size

    def search(self, model: nn.Module, queue: Queue, device: str, print_every: int) -> float:
        if isinstance(model, nn.DataParallel):
            if isinstance(model.module, EncoderDecoderModel):
                if isinstance(model.module.decoder, TransformerDecoder):
                    topk_decoder = BeamTransformerDecoder(
                        model.module.decoder,
                        beam_size=self.k,
                        batch_size=self.batch_size,
                    )
                else:
                    raise ValueError("This model unsupport beam search.")
            elif isinstance(model.module, EncoderModel):
                topk_decoder = BeamCTCDecoder(labels=self.vocab.labels)

            else:
                raise ValueError("This model unsupport beam search.")

            model.module.set_decoder(topk_decoder)
        else:
            if isinstance(model, EncoderDecoderModel):
                if isinstance(model.decoder, TransformerDecoder):
                    topk_decoder = BeamTransformerDecoder(
                        model.module.decoder,
                        beam_size=self.k,
                        batch_size=self.batch_size,
                    )
                else:
                    raise ValueError("This model unsupport beam search.")
            elif isinstance(model, EncoderModel):
                topk_decoder = BeamCTCDecoder(labels=self.vocab.labels)
            else:
                raise ValueError("This model unsupport beam search.")

            model.set_decoder(topk_decoder)
        return super(BeamSearch, self).search(model, queue, device, print_every)
