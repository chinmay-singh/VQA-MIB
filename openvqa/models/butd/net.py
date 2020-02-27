# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.models.butd.tda import TDA
from openvqa.models.butd.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch

import numpy as np
import math

# -------------------------
# ---- Main BUTD Model ----
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

        self.rnn = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = TDA(__C)

        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size= __C.HIDDEN_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # Classification layers
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE,
                                  __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)

        if(self.__C.WITH_ANSWER):

            self.ans_embedding = nn.Embedding(
                num_embeddings=token_size_ans,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

            # Loading the GloVe embedding weights
            if __C.USE_GLOVE:
                if not self.eval_flag:
                    self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb_ans))


            self.ans_rnn = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )
            
            # parameters for storing npy arrays
            self.batch_size = int(__C.SUB_BATCH_SIZE/__C.N_GPU)
            self.num = math.ceil(1000/self.batch_size) #313

            # storing npy arrays
            self.shape = (self.num * self.batch_size, int(__C.HIDDEN_SIZE)) 
            self.z_proj = np.zeros(shape=self.shape)
            self.z_ans = np.zeros(shape=self.shape)
            self.z_fused = np.zeros(shape=self.shape)


    def forward(self, ques_list, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, step, epoch):

        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        self.rnn.flatten_parameters()
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.rnn(lang_feat)

        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        proj_feat = self.backbone(
            lang_feat[:, -1],
            img_feat
        )
        
        self.decoder_gru.flatten_parameters()

        if (self.__C.WITH_ANSWER == False or self.eval_flag == True):
            # use the decoder
            proj_feat, _ = self.decoder_gru(proj_feat.unsqueeze(1))
            proj_feat = proj_feat.squeeze()

            # Classification layers
            proj_feat = self.classifer(proj_feat)
            
            if (self.eval_flag == True and self.__C.WITH_ANSWER == True):
                #hack because test_engine expects multiple returns from net but only uses the first
                return proj_feat, None 

            return proj_feat
        
        ############ WITH ANSWER ##############
        else:

            # --------------------------- #
            # ---- Answer embeddings ---- #
            # --------------------------- #

            ans_feat = self.ans_embedding(ans_ix)
            self.ans_rnn.flatten_parameters()

            ans_feat, _ = self.ans_rnn(ans_feat)
            ans_feat = ans_feat.sum(1)
            ######### or do
            #_, ans_feat = self.ans_rnn(ans_feat) 
            ######## but summing makes more sense when using one word answers only

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
            #debug
            fused_feat = torch.add(torch.mul(u, proj_feat), torch.mul(1-u, ans_feat))

            # --------------------------- #
            # ---- SAVE THE FEATURES ---- #
            # --------------------------- #

            # For calculating Fusion Loss in train_engine
            # also normalize the vectors before calculating loss
            z_proj = F.normalize(proj_feat.clone(), p=2, dim=1)
            z_ans = F.normalize(ans_feat.clone(), p=2, dim=1)
            z_fused = F.normalize(fused_feat.clone(), p=2, dim=1)

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
            proj_feat = self.classifer(proj_feat)
            
            # (batch_size, HIDDEN_SIZE)
            ans_feat, _ = self.decoder_gru(ans_feat.unsqueeze(1))
            ans_feat = ans_feat.squeeze()
            # (batch_size, answer_size)
            ans_feat = self.classifer(ans_feat)
            
            # (batch_size, HIDDEN_SIZE)
            fused_feat, _ = self.decoder_gru(fused_feat.unsqueeze(1))
            fused_feat = fused_feat.squeeze()
            # (batch_size, answer_size)
            fused_feat = self.classifer(fused_feat)

            return proj_feat, ans_feat, fused_feat, z_proj, z_ans, z_fused

 
