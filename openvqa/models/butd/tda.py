# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# based on the implementation in https://github.com/hengyuan-hu/bottom-up-attention-vqa
# ELU is chosen as the activation function in non-linear layers due to
# the experiment results that indicate ELU is better than ReLU in BUTD model.
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch
import math
import cv2
import numpy as np

# ------------------------------
# ----- Weight Normal MLP ------
# ------------------------------

class MLP(nn.Module):
    """
    class for non-linear fully connect network
    """

    def __init__(self, dims, act='ELU', dropout_r=0.0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ------------------------------
# ---Top Down Attention Map ----
# ------------------------------


class AttnMap(nn.Module):
    '''
    implementation of top down attention
    '''
    def __init__(self, __C):
        super(AttnMap, self).__init__()
        self.__C = __C
        self.linear_q = weight_norm(
            nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE), dim=None)
        self.linear_v = weight_norm(
            nn.Linear(__C.IMG_FEAT_SIZE, __C.IMG_FEAT_SIZE), dim=None)
        self.nonlinear = MLP(
            [__C.IMG_FEAT_SIZE + __C.HIDDEN_SIZE, __C.HIDDEN_SIZE], dropout_r=__C.DROPOUT_R)
        self.linear = weight_norm(nn.Linear(__C.HIDDEN_SIZE, 1), dim=None)

    def forward(self, q, v):
        v = self.linear_v(v)
        q = self.linear_q(q)
        logits = self.logits(q, v)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, q, v):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

# ------------------------------
# ---- Attended Joint Map ------
# ------------------------------


class TDA(nn.Module):
    def __init__(self, __C):
        super(TDA, self).__init__()

        self.__C = __C
        self.v_att = AttnMap(__C)
        self.q_net = MLP([__C.HIDDEN_SIZE, __C.HIDDEN_SIZE])
        self.v_net = MLP([__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE])

    def forward(self, q, v, bbox_feat):
        att = self.v_att(q, v)

        if self.__C.USE_NEW_QUESTION == "True":
            print("plotting attention of objects on image")

            img = cv2.imread('test_images/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12) + '.jpg', 3)
            img1 = np.array(img).astype('float64')
            
            bbox_normalized = bbox_feat.squeeze()
            att_squeezed = att.squeeze()
            
            for i in range(bbox_normalized.shape[0]):

                '''
                if float(att_squeezed[i]) < 0.09:
                    continue
                '''
                x1_coordinate = bbox_normalized[i][0]*float(img.shape[1])
                y1_coordinate = bbox_normalized[i][1]*float(img.shape[0])

                x4_coordinate = bbox_normalized[i][2]*float(img.shape[1])
                y4_coordinate = bbox_normalized[i][3]*float(img.shape[0])

                img1[int(y1_coordinate):int(y4_coordinate), int(x1_coordinate):int(x4_coordinate), :] += (255. * float(att_squeezed[i].cpu()))
                img = cv2.rectangle(img, (x1_coordinate, y1_coordinate), (x4_coordinate, y4_coordinate), (255,0,0), 2)        

            cv2.imwrite('test_images/attention/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12) + '.jpg', img1)
            cv2.imwrite('test_images/bounding_box/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12) + '.jpg', img)

        atted_v = (att * v).sum(1)
        q_repr = self.q_net(q)
        v_repr = self.v_net(atted_v)
        joint_repr = q_repr * v_repr
        return joint_repr
