# -*- coding: utf-8 -*-

class Vocabulary(object):
    def __init__(self, *args, **kwargs) -> None:
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def label_to_string(self, labels):
        raise NotImplementedError
