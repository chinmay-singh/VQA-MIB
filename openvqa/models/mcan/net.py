# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Adapter
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.LSTM_NUM_DIRECTIONS * __C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.LSTM_NUM_DIRECTIONS * __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

        # x_shape: (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) 
        # x_mask_shape: (batch, 1, 1, 14)
    def forward(self, x, x_mask):

        # att_shape: (batch, 14, FLAT_GLIMPSES) 
        att = self.mlp(x) 

        # x_mask shape: (batch, 1, 1, 14)
        # x_mask shape after 2 squeeze and unsqueeze: (batch, 14, 1)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )

        # softmax over all the words
        # att_shape: (batch, 14, FLAT_GLIMPSES) 
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                # ith glimpse of every word * x
                # (batch, 14, 1) * (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE) = (batch, 14, NUM_DIRECTIONS*HIDDEN_SIZE)
                # shape of sum: (batch, NUM_DIRECTIONS*HIDDEN_SIZE)
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        # shape: (batch, NUM_DIRECTIONS*HIDDEN_SIZE*FLAT_GLIMPSES)
        x_atted = torch.cat(att_list, dim=1)
        # shape: (batch, FLAT_OUT_SIZE)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, pretrain_emb_ans, token_size_ans):
        super(Net, self).__init__()
        self.__C = __C

        if pretrain_emb_ans is None:
            self.eval_flag = True
        else:
            self.eval_flag = False

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = None
        if (__C.LSTM_NUM_DIRECTIONS is 1):
            self.lstm = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=__C.LSTM_LAYERS,
                batch_first=True,
                bidirectional=False
                )

        elif (__C.LSTM_NUM_DIRECTIONS is 2):
            self.lstm = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=__C.LSTM_LAYERS,
                batch_first=True,
                bidirectional=True
                )

        else:
            sys.exit("LSTM_NUM_DIRECTIONS should be either 1 or 2, current value is: %d" % __C.LSTM_NUM_DIRECTIONS)

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Normalization layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)

        # Decoder GRU and MLP
        self.decoder_gru = nn.GRU(
            input_size= __C.FLAT_OUT_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.decoder_mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size= 2*__C.HIDDEN_SIZE,
            out_size=answer_size,
            dropout_r=0,
            use_relu=True
        )

        # Classification layer
        # self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        if (self.__C.WITH_ANSWER):

            self.ans_embedding = nn.Embedding(
                num_embeddings=token_size_ans,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

            # Loading the GloVe embedding weights
            if __C.USE_GLOVE:
                if not self.eval_flag:
                    self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb_ans))


            if (__C.LSTM_NUM_DIRECTIONS is 1):
                self.lstm_ans = nn.LSTM(
                    input_size=__C.WORD_EMBED_SIZE,
                    hidden_size=__C.HIDDEN_SIZE,
                    num_layers=__C.LSTM_LAYERS,
                    batch_first=True,
                    bidirectional=False
                    )

            elif (__C.LSTM_NUM_DIRECTIONS is 2):
                self.lstm_ans = nn.LSTM(
                    input_size=__C.WORD_EMBED_SIZE,
                    hidden_size=__C.HIDDEN_SIZE,
                    num_layers=__C.LSTM_LAYERS,
                    batch_first=True,
                    bidirectional=True
                )

            self.attflat_ans = AttFlat(__C)
            self.ans_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.fused_norm = LayerNorm(__C.FLAT_OUT_SIZE)

            self.batch_size = int(__C.SUB_BATCH_SIZE/__C.N_GPU)
            self.num = math.ceil(1000/self.batch_size) #313

            # storing npy arrays
            self.shape = (self.num * self.batch_size, int(__C.FLAT_OUT_SIZE)) #(batch, 1024)
            self.z_proj = np.zeros(shape=self.shape) #(batch, 1024)
            self.z_ans = np.zeros(shape=self.shape) #(batch, 1024)
            self.z_fused = np.zeros(shape=self.shape) #(batch, 1024)

    def forward(self, ques_list, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, step, epoch):

        # Visualization of top 3 objects on orignal image
        if self.__C.USE_NEW_QUESTION == "True":
            img = np.array(Image.open('./COCO_test2015_000000126672.jpg'), dtype=np.uint8)
            print("orignal image shape: ", img.shape)  
            fig,ax = plt.subplots(1)

            ax.imshow(img)

            temp = np.array(bbox_feat.cpu())[0]
            j = 0
            polygons = []
            for i in temp:
                if j > 3:
                    break
                x1_coordinate = i[0]*img.shape[1]
                y1_coordinate = i[1]*img.shape[0]
                x4_coordinate = i[2]*img.shape[1] 
                y4_coordinate = i[3]*img.shape[0]

                width = x4_coordinate - x1_coordinate
                height = y4_coordinate - y1_coordinate

                x2_coordinate = x4_coordinate
                y2_coordinate = y1_coordinate
                x3_coordinate = x1_coordinate
                y3_coordinate = y4_coordinate
               
                poly = [[x1_coordinate, y1_coordinate], [x2_coordinate, y2_coordinate], [x3_coordinate, y3_coordinate], [x4_coordinate, y4_coordinate]]
                np_poly = np.array(poly).reshape(4,2)
#                polygons.append(Polygon(np_poly))
                polygons.append(Rectangle((x1_coordinate, y1_coordinate), width, height))
                j += 1

            p = PatchCollection(polygons, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_collection(p)

            print("saving image of plotted object")
            plt.savefig("./plotted_objects0.jpg") 

        # Pre-process Language Feature
        # Returns (batch, 1, 1, 14)
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))

        # Returns (batch, 14, WORD_EMBED_SIZE)
        lang_feat = self.embedding(ques_ix)

        # h_n and c_n are returned as a tuple, ignored in _
        # h_n (batch, NUM_DIRECTIONS * NUM_LAYERS, HIDDEN_SIZE)
        # c_n (batch, NUM_DIRECTIONS * NUM_LAYERS, HIDDEN_SIZE)
        # Input (batch, 14, input_size_of_lstm = WORD_EMBED_SIZE)
        # output (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE)
        self.lstm.flatten_parameters()
        lang_feat, _ = self.lstm(lang_feat)


        # Returns (batch, 100, NUM_DIRECTIONS * HIDDEN_SIZE), (batch, 1, 1, 100)
        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat) 

        # Backbone Framework 
        # (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE) (batch, 100, NUM_DIRECTIONS * HIDDEN_SIZE)
        lang_feat, img_feat = self.backbone(
            ques_list,    
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        # shape: (batch, FLAT_OUT_SIZE)
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        # shape: (batch, FLAT_OUT_SIZE)
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        # Here, lang_feat and img_feat have been multiplied by matrices in the last step to bring them to a common space so that they can be added
        # shape: (batch, FLAT_OUT_SIZE)
        proj_feat = lang_feat + img_feat

        self.decoder_gru.flatten_parameters()

        if (self.__C.WITH_ANSWER == False or self.eval_flag == True):

            # Normalization
            # (batch, FLAT_OUT_SIZE)
            proj_feat = self.proj_norm(proj_feat) 

            # Decoder
            proj_feat, _ = self.decoder_gru(proj_feat.unsqueeze(1))
            proj_feat = proj_feat.squeeze()
            # (batch_size, answer_size)
            proj_feat = self.decoder_mlp(proj_feat)
     
            if (self.eval_flag == True and self.__C.WITH_ANSWER == True):
                #hack because test_engine expects multiple returns from net but only uses the first
                return proj_feat, None 

            return proj_feat

        ####### WITH ANSWER ########
        else:

            # --------------------------- #
            # ---- Answer embeddings ---- #
            # --------------------------- #

            ans_feat_mask = make_mask(ans_ix.unsqueeze(2))
            ans_feat = self.ans_embedding(ans_ix)
            self.lstm_ans.flatten_parameters()

            # output (batch, 4, NUM_DIRECTIONS * HIDDEN_SIZE)
            ans_feat, _ = self.lstm_ans(ans_feat)

            # shape: (batch, FLAT_OUT_SIZE)
            ans_feat = self.attflat_ans(
                ans_feat,
                ans_feat_mask
            )

            # ---------------------- #
            # ---- Adding noise ---- #
            # ---------------------- #

            # randomly sample a number 'u' between zero and one
            u = torch.rand(1).cuda()

            proj_noise = self.__C.PROJ_STDDEV * torch.randn(proj_feat.shape).cuda()
            ans_noise = self.__C.ANS_STDDEV * torch.randn(ans_feat.shape).cuda()
            
            ans_feat += ans_noise
            proj_feat += proj_noise

            # now we can fuse the vector
            # (batch_size, FLAT_OUT_SIZE)
            fused_feat = torch.add(torch.mul(u, proj_feat), torch.mul(1-u, ans_feat))

            # ------------------- #
            # ---- NORMALIZE ---- #
            # ------------------- #

            proj_feat = self.proj_norm(proj_feat) 
            ans_feat = self.ans_norm(ans_feat)
            fused_feat = self.fused_norm(fused_feat)

            # --------------------------- #
            # ---- SAVE THE FEATURES ---- #
            # --------------------------- #

            # For calculating Fusion Loss in train_engine
            z_proj = proj_feat.clone().detach()
            z_ans = ans_feat.clone().detach()
            z_fused = fused_feat.clone().detach()

            if (step < self.num):
                self.z_proj[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = proj_feat.clone().detach().cpu().numpy()
                self.z_ans[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = ans_feat.clone().detach().cpu().numpy()
                self.z_fused[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = fused_feat.clone().detach().cpu().numpy()


            elif (step == self.num):
                np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_proj_' + str(epoch) + '.npy', self.z_proj)
                np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_ans_' + str(epoch) + '.npy', self.z_ans)
                np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_fused_' + str(epoch) + '.npy', self.z_fused)

            elif (step == (self.num + 1)):
                self.z_proj = np.zeros(shape=self.shape)
                self.z_ans = np.zeros(shape=self.shape)
                self.z_fused = np.zeros(shape=self.shape)

            # ----------------- #
            # ---- DECODER ---- #
            # ----------------- #

            # (batch_size, HIDDEN_SIZE)
            proj_feat, _ = self.decoder_gru(proj_feat.unsqueeze(1))
            proj_feat = proj_feat.squeeze()
            # (batch_size, answer_size)
            proj_feat = self.decoder_mlp(proj_feat)
            
            # (batch_size, HIDDEN_SIZE)
            ans_feat, _ = self.decoder_gru(ans_feat.unsqueeze(1))
            ans_feat = ans_feat.squeeze()
            # (batch_size, answer_size)
            ans_feat = self.decoder_mlp(ans_feat)
            
            # (batch_size, HIDDEN_SIZE)
            fused_feat, _ = self.decoder_gru(fused_feat.unsqueeze(1))
            fused_feat = fused_feat.squeeze()
            # (batch_size, answer_size)
            fused_feat = self.decoder_mlp(fused_feat)

            return proj_feat, ans_feat, fused_feat, z_proj, z_ans, z_fused
