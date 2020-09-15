# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, torch, random
import torch.utils.data as Data
import torch.nn as nn
from openvqa.utils.feat_filter import feat_filter

class BaseDataSet(Data.Dataset):
    def __init__(self):
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None

        self.data_size = None
        self.token_size = None
        self.ans_size = None


    def load_ques_ans(self, idx):
        raise NotImplementedError()


    def load_img_feats(self, idx, iid):
        raise NotImplementedError()


    def __getitem__(self, idx):
        """
        Returned keys: ['ques_ix_iter', 'ans_iter', 'iid', 'question', 'augmented_answer', 
        'answer', 'question_type', 'frcn_feat_iter', 'grid_feat_iter', 'bbox_feat_iter']

        """

        """
        From load_ques_ans
        ret_dict = {
            'ques_ix_iter': ques_ix_iter,
            'ans_iter': ans_iter,
            'iid': iid,
            'question': qu_proc,
            'augmented_answer': aug_ans_proc,
            'answer': an_proc,
            'question_type': ty_proc
        }
        """
        ret_dict = self.load_ques_ans(idx)
        #ret_dict['ques_ix_iter'] = torch.from_numpy(ret_dict['ques_ix_iter'])
        #ret_dict['ans_iter'] = torch.from_numpy(ret_dict['ans_iter'])


        iid = ret_dict['iid']

        frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)
        ret_dict['frcn_feat_iter'] = torch.from_numpy(frcn_feat_iter)
        ret_dict['grid_feat_iter'] = torch.from_numpy(grid_feat_iter)
        ret_dict['bbox_feat_iter'] = torch.from_numpy(bbox_feat_iter)

        # convert remaining numpy arrays to torch tensors
        for k, v in ret_dict.items():
            if isinstance(v, (np.ndarray, np.generic)):
                ret_dict[k] = torch.from_numpy(v)

        return ret_dict


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)


class BaseAdapter(nn.Module):
    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        if self.__C.DATASET in ['vqa']:
            self.vqa_init(__C)

        elif self.__C.DATASET in ['gqa']:
            self.gqa_init(__C)

        elif self.__C.DATASET in ['clevr']:
            self.clevr_init(__C)

        else:
            exit(-1)

        # eval('self.' + __C.DATASET + '_init()')

    def vqa_init(self, __C):
        raise NotImplementedError()

    def gqa_init(self, __C):
        raise NotImplementedError()

    def clevr_init(self, __C):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(self.__C.DATASET, frcn_feat, grid_feat, bbox_feat)

        if self.__C.DATASET in ['vqa']:
            return self.vqa_forward(feat_dict)

        elif self.__C.DATASET in ['gqa']:
            return self.gqa_forward(feat_dict)

        elif self.__C.DATASET in ['clevr']:
            return self.clevr_forward(feat_dict)

        else:
            exit(-1)

    def vqa_forward(self, feat_dict):
        raise NotImplementedError()

    def gqa_forward(self, feat_dict):
        raise NotImplementedError()

    def clevr_forward(self, feat_dict):
        raise NotImplementedError()

