# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS, __C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS, __C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS, __C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS, __C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    # shape of v is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
    def forward(self, v, k, q, mask, visualization):
        n_batches = q.size(0)

        # shape of v is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
        # shape of v after linear is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
        # shape of v after view is : (batch, 14, MULTI_HEAD,  NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
        # shape of v after transpose is : (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int((self.__C.HIDDEN_SIZE * self.__C.LSTM_NUM_DIRECTIONS)/ self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        # shape of k after transpose is : (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int((self.__C.HIDDEN_SIZE * self.__C.LSTM_NUM_DIRECTIONS)/ self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        # shape of q after transpose is : (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int((self.__C.HIDDEN_SIZE * self.__C.LSTM_NUM_DIRECTIONS)/ self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        # shape of atted: (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD)
        atted = self.att(v, k, q, mask, visualization)

        # shape of atted: (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD)
        # shape of atted after transpose: (batch, 14, MULTI_HEAD, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD)
        # shape of atted: (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE * self.__C.LSTM_NUM_DIRECTIONS
        )

        # shape of atted: (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
        atted = self.linear_merge(atted)

        # shape of atted: (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
        return atted

    # shape of q is : (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
    def att(self, value, key, query, mask, visualization):

        # d_k = NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD
        d_k = query.size(-1)

        # shape of q is : (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
        # shape of k after transpose is : (batch, MULTI_HEAD, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD, 14) 
        # shape of scores : (batch, MULTI_HEAD, 14, 14) 
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)


        # shape of scores : (batch, MULTI_HEAD, 14, 14) 
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # shape of att_map : (batch, MULTI_HEAD, 14, 14) 
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        
        if self.__C.USE_NEW_QUESTION == "True" and visualization == True:
#            print("Making Visualizations: ", att_map)
            print("####################################")
            print(att_map.shape)
            temp = att_map.reshape((att_map.shape[1], att_map.shape[0], att_map.shape[2], att_map.shape[3]))
            temp = temp.cpu()

            i = 0
            for x_ in temp:
                x_ = np.array(x_)
                fig, ax = plt.subplots()
                im = ax.imshow(x_[0])
                for p in range(x_[0].shape[0]):
                    for q in range(x_[0].shape[1]):
                        text = ax.text(q, p, round(x_[0][p, q], 2), ha = "center", va = "center", color = "w")

                fig.tight_layout()
                plt.savefig(self.__C.RESULT_PATH + "testing" + str(i) + ".jpg")
                i += 1
                plt.close()

        # shape of att_map : (batch, MULTI_HEAD, 14, 14) 
        # shape of v : (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD) 
        # shape of return value: (batch, MULTI_HEAD, 14, NUM_DIRECTIONS*HIDDEN_SIZE/ MULTI_HEAD)
        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)

    # shape of y is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
    # shape of y_mask is : (batch, 1, 1, 14) 
    def forward(self, y, y_mask, visualization):
        y = self.norm1(y + self.dropout1(
            # shape after: (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
            self.mhatt(y, y, y, y_mask, visualization=visualization)
        ))

        y = self.norm2(y + self.dropout2(
            # shape after: (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
            self.ffn(y)
        ))

        # shape of return: (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE * __C.LSTM_NUM_DIRECTIONS)

    # shape of y is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
    # shape of y_mask is : (batch, 1, 1, 14) 
    # shape of x is : (batch, 100, NUM_DIRECTIONS * HIDDEN_SIZE)
    # shape of x_mask is : (batch, 1, 1, 100)
    def forward(self, x, y, x_mask, y_mask, visualization):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask, visualization=visualization)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask, visualization=visualization)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    # shape of y is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
    # shape of y_mask is : (batch, 1, 1, 14) 
    # shape of x is : (batch, 100, NUM_DIRECTIONS * HIDDEN_SIZE)
    # shape of x_mask is : (batch, 1, 1, 100)

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:

            # shape of y is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
            y = enc(y, y_mask, visualization = True)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors

        for dec in self.dec_list:
            # shape of y is : (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
            x = dec(x, y, x_mask, y_mask, visualization = False)

        return y, x
