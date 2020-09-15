# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.models.mfb.mfb import CoAtt
from openvqa.models.mfb.adapter import Adapter
import torch
import torch.nn as nn

from collections import defaultdict

# -------------------------------------------------------
# ---- Main MFB/MFH model with Co-Attention Learning ----
# -------------------------------------------------------


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.adapter = Adapter(__C)

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.LSTM_OUT_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.dropout_lstm = nn.Dropout(__C.DROPOUT_R)
        self.backbone = CoAtt(__C)

        if __C.HIGH_ORDER:      # MFH
            self.proj = nn.Linear(2*__C.MFB_O, answer_size)
        else:                   # MFB
            self.proj = nn.Linear(__C.MFB_O, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, **kwargs):

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)  # (N, C, FRCN_FEAT_SIZE)

        # Pre-process Language Feature
        ques_feat = self.embedding(ques_ix)     # (N, T, WORD_EMBED_SIZE)
        ques_feat = self.dropout(ques_feat)
        ques_feat, _ = self.lstm(ques_feat)     # (N, T, LSTM_OUT_SIZE)
        ques_feat = self.dropout_lstm(ques_feat)

        text_ret = defaultdict(list)
        img_ret = defaultdict(list)

        z = self.backbone(
            img_feat,
            ques_feat,
            text_ret=text_ret,
            img_ret=img_ret
        )  # MFH:(N, 2*O) / MFB:(N, O)
        proj_feat = self.proj(z)                # (N, answer_size)

        return_dict = {
            "proj_feat": proj_feat,
            "img": img_ret,
            "text": text_ret
        }
        return return_dict

