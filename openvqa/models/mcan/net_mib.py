# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Adapter
from openvqa.models.mcan.net import Net as Branch

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb_ques: np.ndarray, pretrained_emb_ans: np.ndarray, token_size: int, answer_size: int, **kwargs: dict):
        """
        Class for pretraining the MCAN with answers

        Args:
            __C (Config): Config file
            pretrained_emb_ques (np.ndarray): A numpy array to initialise the embeddings of question branch from
            pretrained_emb_ans (np.ndarray): A numpy array to initialise the embeddings of answer branch from
            token_size (int): Number of words in the vocabulary
            answer_size (int): Number of answer classes
        """
        super(Net, self).__init__()

        self.__C = __C

        self.kwargs = kwargs

        self.question_branch = Branch(__C, pretrained_emb=pretrained_emb_ques, token_size=token_size, answer_size=answer_size, kwargs=kwargs)
        self.answer_branch = Branch(__C, pretrained_emb=pretrained_emb_ans, token_size=token_size, answer_size=answer_size, kwargs=kwargs)

        # Freezing the answer branch
        for name, param in self.answer_branch.named_parameters():
            param.requires_grad = False
        
        self.answer_branch.eval()

    def return_dict(self):
        """
        Returns the format of the dict that the forward function returns

        """

        return {
            "ques": self.question_branch.return_dict(),
            "ans": self.answer_branch.return_dict()
        }

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, **kwargs):

        ques = self.question_branch(frcn_feat, grid_feat, bbox_feat, ques_ix)
        ans = self.answer_branch(frcn_feat, grid_feat, bbox_feat, ans_ix)

        return_dict = {}
        return_dict['ques'] = ques
        return_dict['ans'] = ans

        return return_dict

