# https://github.com/Mathux/TEMOS/blob/master/temos/model/textencoder/distilbert_actor.py
from model.distilbert import DistilbertEncoderBase
import torch

from typing import List, Union
from torch import nn, Tensor


class DistilbertEncoder(DistilbertEncoderBase):
    def __init__(self, modelpath: str, finetune: bool = False) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)
        # encoded_dim = self.text_encoded_dim
        
    def forward(self, texts: List[str], max_length: int = 20) -> Tensor:
        text_encoded, mask = self.get_last_hidden_state(texts, max_length,return_mask=True)
        return text_encoded
