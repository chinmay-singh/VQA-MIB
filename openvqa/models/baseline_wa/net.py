from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
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

    def forward(self, x, x_mask):
        att = self.mlp(x) # (64, 100, 1)

        # x_mask shape: (batch, 1, 1, 100)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )

        att = F.softmax(att, dim=1)

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


#Generator



    

class Net(nn.Module):
    
    '''
    init function has 3 input parameters-

    pretrained_emb: corresponds to the GloVe embedding features for the question
    token_size: corresponds to the number of all dataset words
    answer_size: corresponds to the number of classes for prediction
    '''
    def __init__(self, __C, pretrained_emb, token_size, answer_size, pretrain_emb_ans, token_size_ans, noise_sigma = 0.1):
        super(Net, self).__init__()
        if pretrain_emb_ans is None:
            self.eval_flag = True
        else:
            self.eval_flag = False

        self.__C = __C

        self.decoder_mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size= 2*__C.HIDDEN_SIZE,
            out_size=answer_size,
            dropout_r=0,
            use_relu=True
        )

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.ans_embedding = nn.Embedding(
            num_embeddings=token_size_ans,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

            #Edits
            if not self.eval_flag:
                self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb_ans))
            #End of Edits

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # Generator

        self.gru_gen = nn.GRU(
            input_size= __C.FLAT_OUT_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # End of Generator

        self.ans_lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.attflat_ans = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        # Answer Classification layers
        self.ans_proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.ans_proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        
        # create the noise vector std
        self.noise_sigma = noise_sigma

        self.batch_size = int(__C.SUB_BATCH_SIZE/__C.N_GPU)
        self.num = math.ceil(1000/self.batch_size) #313

        # storing npy arrays
        self.shape = (self.num * self.batch_size, int(__C.FLAT_OUT_SIZE)) #(10016, 1024)
        self.z_proj = np.zeros(shape=self.shape) #(10016, 1024)
        self.z_ans = np.zeros(shape=self.shape) #(10016, 1024)
        self.z_fused = np.zeros(shape=self.shape) #(10016, 1024)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, step, epoch):

        step = int(step)


        '''
        We need to implement this function differently for the train and tes
        therefore, just move the answer feature processing into another file
        and call that in the forward function based on the if and else conditions
        on whether the mode is train or test
        '''

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix) # (batch, 14, 300)

        self.lstm.flatten_parameters()
        lang_feat, _ = self.lstm(lang_feat) # (batch, 14, 512)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat) # (batch, 100, 512), (batch, 1, 1, 100)

       # Flatten to vector
        # (batch, 1024)
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        # (batch, 1024)
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat # (batch, 1024)

        # Pre-process answer Feature
        ans_feat = self.ans_embedding(ans_ix) # (batch, 4, 300)

        self.ans_lstm.flatten_parameters()
        ans_feat, _ = self.ans_lstm(ans_feat) # (batch, 4, 512)
        ans_feat_mask = torch.randn(ans_feat.shape[0], 1, 1, ans_feat.shape[1]).bool().cuda()

        # Flatten to vector
        # (batch, 1024)
        ans_feat = self.attflat_ans(
            ans_feat,
            ans_feat_mask
        )

        # Add noise to both encoded representations
        # self.noise_sigma is to be passed
        noise_vec = self.__C.PROJ_STDDEV * torch.randn(proj_feat.shape).cuda()
        ans_noise_vec = self.__C.ANS_STDDEV * torch.randn(ans_feat.shape).cuda()


        if not self.eval_flag:
            ans_feat += ans_noise_vec
            proj_feat += noise_vec

        # randomly sample a number 'u' between zero and one
        u = torch.rand(1).cuda()

        # now we can fuse the vector
        if not self.eval_flag:
            # (batch_size, 1024)
            fused_feat = torch.add(torch.mul(u, proj_feat), torch.mul(1-u, ans_feat))
        else:
            fused_feat = proj_feat

        # For calculating Fusion Loss in train_engine
        z_proj = proj_feat.clone().detach()
        z_ans = ans_feat.clone().detach()
        z_fused = fused_feat.clone().detach()

        # Save the three features
        if (step < self.num and not self.eval_flag):
            self.z_proj[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = proj_feat.clone().detach().cpu().numpy()
            self.z_ans[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = ans_feat.clone().detach().cpu().numpy()
            self.z_fused[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = fused_feat.clone().detach().cpu().numpy()


        elif (step == self.num and not self.eval_flag):
            np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_proj_' + str(epoch) + '.npy', self.z_proj)
            np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_ans_' + str(epoch) + '.npy', self.z_ans)
            np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_fused_' + str(epoch) + '.npy', self.z_fused)

        elif (step == (self.num + 1) and not self.eval_flag):
            self.z_proj = np.zeros(shape=self.shape)
            self.z_ans = np.zeros(shape=self.shape)
            self.z_fused = np.zeros(shape=self.shape)

        # DECODER
        self.gru_gen.flatten_parameters()

        # (batch_size, 512)
        proj_feat, _ = self.gru_gen(proj_feat.unsqueeze(1))
        proj_feat = proj_feat.squeeze()
        # (batch_size, answer_size)
        proj_feat = self.decoder_mlp(proj_feat)
        
        # (batch_size, 512)
        ans_feat, _ = self.gru_gen(ans_feat.unsqueeze(1))
        ans_feat = ans_feat.squeeze()
        # (batch_size, answer_size)
        ans_feat = self.decoder_mlp(ans_feat)
        
        # (batch_size, 512)
        fused_feat, _ = self.gru_gen(fused_feat.unsqueeze(1))
        fused_feat = fused_feat.squeeze()
        # (batch_size, answer_size)
        fused_feat = self.decoder_mlp(fused_feat)

        return proj_feat, ans_feat, fused_feat, z_proj, z_ans, z_fused
