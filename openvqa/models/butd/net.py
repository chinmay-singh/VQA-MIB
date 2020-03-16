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

        if __C.USING_PRETRAINED:

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

            '''
            # Decoder GRU
            self.decoder_gru = nn.GRU(
                input_size= __C.HIDDEN_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )
            '''
        # Classification layers
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE,
                                  __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)

        if(self.__C.WITH_ANSWER or not self.__C.USING_PRETRAINED):

            self.ans_embedding = nn.Embedding(
                num_embeddings=token_size_ans,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

            # Loading the GloVe embedding weights
            if __C.USE_GLOVE:
                if not __C.USING_PRETRAINED or not self.eval_flag:
                    self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb_ans))


            self.ans_rnn = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )
            
            self.ans_adapter = Adapter(__C)

            self.ans_backbone = TDA(__C)
            
            if (self.__C.WITH_ANSWER):
                # parameters for storing npy arrays
                self.batch_size = int(__C.SUB_BATCH_SIZE/__C.N_GPU)
                self.num = math.ceil(1000/self.batch_size) #313

                # storing npy arrays
                self.shape = (self.num * self.batch_size, int(__C.HIDDEN_SIZE)) 
                self.z_proj = np.zeros(shape=self.shape)
                self.z_ans = np.zeros(shape=self.shape)
                self.z_fused = np.zeros(shape=self.shape)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, step, epoch):


        if not self.__C.USING_PRETRAINED:
            
            # Pre-process the answer features
            self.ans_rnn.flatten_parameters()
            ans_lang_feat = self.ans_embedding(ans_ix)
            ans_lang_feat, _ = self.ans_rnn(ans_lang_feat)

            ans_img_feat, _ = self.ans_adapter(frcn_feat, grid_feat, bbox_feat)

            ans_proj_feat = self.ans_backbone(ans_lang_feat[:, -1], ans_img_feat)

            ans_proj_feat = self.classifer(ans_proj_feat)

            return ans_proj_feat

        else:

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

            if (self.__C.WITH_ANSWER == False or self.eval_flag == True):
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
                # Pre-process the answer features
                self.ans_rnn.flatten_parameters()
                ans_lang_feat = self.ans_embedding(ans_ix)
                ans_lang_feat, _ = self.ans_rnn(ans_lang_feat)

                ans_img_feat, _ = self.ans_adapter(frcn_feat, grid_feat, bbox_feat)

                ans_feat = self.ans_backbone(ans_lang_feat[:, -1], ans_img_feat)

                
               
                # ---------------------- #
                # ---- Adding noise ---- #
                # ---------------------- #

                # randomly sample a number 'u' between zero and one
                u = torch.rand(1).cuda()

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

                
                proj_feat = self.classifer(proj_feat)
                
                ans_feat = self.classifer(ans_feat)
                
                fused_feat = self.classifer(fused_feat)

                return proj_feat, ans_feat, fused_feat, z_proj, z_ans, z_fused

 
