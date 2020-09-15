# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from collections import defaultdict

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------


class AttFlat(nn.Module):
    """
    This module returns two things

    Returns:
        x_attend : The attended modality
    """
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask, ret: defaultdict(list)):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)
        ret['att_flat'].append(att)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb: np.ndarray, token_size: int, answer_size: int, **kwargs: dict):
        """
        Class for pretraining the MCAN with answers

        Args:
            __C (Config): Config file
            pretrained_emb (np.ndarray): A numpy array to initialise the embeddings from
            token_size (int): Number of words in the vocabulary
            answer_size (int): Number of answer classes
        """
        super(Net, self).__init__()

        self.__C = __C

        self.kwargs = kwargs

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb)) # type: ignore

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def return_dict(self) -> dict:
        """
        Returns the format of the dict that the forward function returns

        Returns:
        {
            "proj_feat" : [batch, answer_size]
            "img" : {
                "sa": list(Tensor([batch, n_heads, num_objects, num_objects])),
                "ca": list(Tensor([batch, n_heads, num_objects, num_words])),
                "att_flat": list(Tensor([batch, num_objects, 1]))
            }
            "text" : {
                "sa": list(Tensor([batch, n_heads, num_words, num_words])),
                "att_flat": list(Tensor([batch, num_words, 1]))
            }
        }

        """
        return {
            "proj_feat": "[batch, answer_size]",
            "img": {
                "sa": "list(Tensor([batch, n_heads, num_objects, num_objects]))",
                "ca": "list(Tensor([batch, n_heads, num_objects, num_words]))",
                "att_flat": "list(Tensor([batch, num_objects, 1]))"
            },
            "text": {
                "sa": "list(Tensor([batch, n_heads, num_words, num_words]))",
                "att_flat": "list(Tensor([batch, num_words, 1]))"
            }
        }
    
    def forward(self, frcn_feat, grid_feat, bbox_feat, text_ix, **kwargs):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(text_ix.unsqueeze(2))
        lang_feat = self.embedding(text_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        text_ret = defaultdict(list)
        img_ret = defaultdict(list)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            text_ret=text_ret,
            img_ret=img_ret
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask,
            text_ret
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask,
            img_ret
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return_dict = {
            "proj_feat": proj_feat,
            "img": img_ret,
            "text": text_ret
        }
        return return_dict

