import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.token_len = configs.token_len

        # Decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(1, 512, 'timeF', 'h',
                                                  0.1)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, 3, attention_dropout=0.1),
                        512, 8),
                    512,
                    2048,
                    moving_avg=25,
                    dropout=0.1,
                    activation='gelu'
                ) for l in range(2)
            ],
            norm_layer=my_Layernorm(512)
        )
        # Decoder
  
        self.dec_embedding = DataEmbedding_wo_pos(1, 512, 'timeF', 'a',
                                                    0.1)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, 3, attention_dropout=0.1,
                                        output_attention=False),
                        512, 8),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, 3, attention_dropout=0.1,
                                        output_attention=False),
                        512, 8),
                    512,
                    1,
                    2048,
                    moving_avg=25,
                    dropout=0.1,
                    activation='gelu',
                )
                for l in range(1)
            ],
            norm_layer=my_Layernorm(512),
            projection=nn.Linear(512, 1, bias=True)
        )


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.token_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.token_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(x_enc, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
        #                                          trend=trend_init)
        # final
        # dec_out = trend_part + seasonal_part
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc, x_enc, x_mark_dec)
        return dec_out[:, -self.token_len:, :]  # [B, L, D]
