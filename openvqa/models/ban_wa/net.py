# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.ban_wa.ban_wa import BAN
from openvqa.models.ban_wa.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch
import math
import numpy as np

# -------------------------
# ---- Main BAN Model With Answers----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, pretrain_emb_ans, token_size_ans, noise_sigma = 0.1):
        super(Net, self).__init__()

        if pretrain_emb_ans is None:
            self.eval_flag = True
            print("\n----------------\nEval time, eval_flag = true\n----------------------\n")
        else:
            self.eval_flag = False
            print("\n----------------\nTrain time, eval_flag = false\n----------------------\n")
        
        self.__C = __C

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

        self.rnn = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        
        self.ans_rnn = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        
        # Generator

        self.gru_gen = nn.GRU(
            input_size= __C.HIDDEN_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # End of Generator
        self.adapter = Adapter(__C)
        self.backbone = BAN(__C)


        # Classification layers after generator
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE, __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifier = nn.Sequential(*layers)
        
        # create the noise vector std
        self.noise_sigma = noise_sigma
        
        # parameters for saving numpy arrays
        self.batch_size = int(__C.SUB_BATCH_SIZE)
        self.num = math.ceil(10000/self.batch_size) #313

        # storing npy arrays
        self.shape = (self.num * self.batch_size, int(__C.HIDDEN_SIZE)) #(10016, 1024) changed flat out size to hidden size
        self.z_proj = np.zeros(shape=self.shape) #(10016, 1024)
        self.z_ans = np.zeros(shape=self.shape) #(10016, 1024)
        self.z_fused = np.zeros(shape=self.shape) #(10016, 1024)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, step, epoch):

        step = int(step)

        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        self.rnn.flatten_parameters()
        lang_feat, _ = self.rnn(lang_feat)
        
        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        proj_feat = self.backbone(
            lang_feat,
            img_feat
        )

        # sum the lang+img features along dimesion 1
        proj_feat = proj_feat.sum(1)
        
        # create a noise vector
        noise_vec = self.noise_sigma*torch.randn(proj_feat.shape).cuda()

        # ans features
        ans_feat = self.ans_embedding(ans_ix)
        self.ans_rnn.flatten_parameters()
        ans_feat, _ = self.ans_rnn(ans_feat)
        ans_feat = ans_feat.sum(1)
        ans_noise_vec = self.noise_sigma*torch.randn(ans_feat.shape).cuda()
        
        # add different noise to ans_feat but only at training time
        if not self.eval_flag:
            assert ans_feat.shape == proj_feat.shape, "ans_feat: {} and proj_feat: {} shapes do not match".format(ans_feat.shape, proj_feat.shape)
            ans_feat += noise_vec
      
            # add the noise to lang+img features
            proj_feat += noise_vec
           
        # randomly sample a number 'u' between zero and one
        u = torch.rand(1).cuda() 
       
        # now we can fuse the vector
        if not self.eval_flag:
            fused_feat = torch.add(torch.mul(u, proj_feat), torch.mul(1-u, ans_feat))
        else:
            fused_feat = proj_feat
        
        # Save the three features
        if (step < self.num and not self.eval_flag):
            self.z_proj[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = proj_feat.clone().detach().cpu().numpy()
            self.z_ans[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = ans_feat.clone().detach().cpu().numpy()
            self.z_fused[ (step * self.batch_size) : ((step+1) * self.batch_size) ] = fused_feat.clone().detach().cpu().numpy()

        elif (step == self.num and not self.eval_flag):
            np.save('/mnt/sdb/yash/openvqa/saved/ban_wa/z_proj_' + str(epoch) + '.npy', self.z_proj)
            np.save('/mnt/sdb/yash/openvqa/saved/ban_wa/z_ans_' + str(epoch) + '.npy', self.z_ans)
            np.save('/mnt/sdb/yash/openvqa/saved/ban_wa/z_fused_' + str(epoch) + '.npy', self.z_fused)

        elif (step == (self.num + 1) and not self.eval_flag):
            self.z_proj = np.zeros(shape=self.shape)
            self.z_ans = np.zeros(shape=self.shape)
            self.z_fused = np.zeros(shape=self.shape)


        # add the decoder
        # DECODER
        self.gru_gen.flatten_parameters()

        # (batch_size, 512)
        proj_feat, _ = self.gru_gen(proj_feat.unsqueeze(1))
        proj_feat = proj_feat.squeeze()
        # (batch_size, answer_size)
        proj_feat = self.classifier(proj_feat)
        
        # (batch_size, 512)
        ans_feat, _ = self.gru_gen(ans_feat.unsqueeze(1))
        ans_feat = ans_feat.squeeze()
        # (batch_size, answer_size)
        ans_feat = self.classifier(ans_feat)
        
        # (batch_size, 512)
        fused_feat, _ = self.gru_gen(fused_feat.unsqueeze(1))
        fused_feat = fused_feat.squeeze()
        # (batch_size, answer_size)
        fused_feat = self.classifier(fused_feat)
        
        return proj_feat, ans_feat, fused_feat
