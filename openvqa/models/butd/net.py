# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# Modified at FrostLabs
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
    def __init__(self, __C, pretrained_emb, token_size, answer_size, pretrain_emb_ans, token_size_ans, training=True):
        super(Net, self).__init__()
        self.__C = __C
        self.training = training

        """
        Modes of training that this net should support
        1. The original VQA model
        2. Training only the answer (teacher) branch
        3. Training both the question (student) and answer (teacher) branch simultaneously
        4. Using the pre-trained answer branch and training the question branch
        """

        if __C.TRAINING_MODE in ['original', 'simultaneous_qa', 'pretrained_ans']:

            """
            The word embedding layer
            """
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

            # Loading the GloVe embedding weights
            if __C.USE_GLOVE:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

            """
            The lstm for processing the question
            """
            self.rnn = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )

            """
            Adapter and the backbone
            """

            self.adapter = Adapter(__C)
            self.backbone = TDA(__C)

        if __C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans'] and self.training: 

            """
            The word embedding layer
            """
            self.ans_embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

            # Loading the GloVe embedding weights
            if __C.USE_GLOVE:
                self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrained_emb_ans))

            """
            The lstm for processing the question
            """
            self.ans_rnn = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )

            """
            Adapter and the backbone
            """

            self.ans_adapter = Adapter(__C)
            self.ans_backbone = TDA(__C)

            # Freeze the answer branch if it is pre-trained
            if __C.TRAINING_MODE == 'pretrained_ans':
                for params in self.ans_embedding.parameters():
                    params.requires_grad = False
                for params in self.ans_rnn.parameters():
                    params.requires_grad = False
                for params in self.ans_backbone.parameters():
                    params.requires_grad = False
                for params in self.ans_adapter.parameters():
                    params.requires_grad = False

        if __C.TRAINING_MODE == 'pretraining_ans':

            """
            The word embedding layer
            """
            self.ans_embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

            # Loading the GloVe embedding weights
            if __C.USE_GLOVE:
                self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrained_emb_ans))

            """
            The lstm for processing the question
            """
            self.ans_rnn = nn.LSTM(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )

            """
            Adapter and the backbone
            """

            self.ans_adapter = Adapter(__C)
            self.ans_backbone = TDA(__C)
            
        """
        Classification layers remain same for all training modes
        """
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE, __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFIER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifier = nn.Sequential(*layers)

        if __C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans']:
            # parameters for storing npy arrays
            self.batch_size = int(__C.SUB_BATCH_SIZE/__C.N_GPU)
            self.num = math.ceil(1000/self.batch_size) #313

            # storing npy arrays
            self.shape = (self.num * self.batch_size, int(__C.HIDDEN_SIZE)) 
            self.z_ques_branch = np.zeros(shape=self.shape)
            self.z_ans_branch = np.zeros(shape=self.shape)
            self.z_fused = np.zeros(shape=self.shape)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix, step, epoch):

        if self.__C.TRAINING_MODE in ['original', 'simultaneous_qa', 'pretrained_ans']:
            """
            Apply the image adapter
            """

            img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)

            """
            Process the language features
            """
            self.rnn.flatten_parameters()
            ques_feat = self.embedding(ques_ix)
            ques_feat = self.rnn(ques_feat)
            
            """
            Pass both image and ques features through backbone
            """
            ques_branch_feat = self.backbone(img_feat, ques_feat)

        if self.__C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans'] and self.training:
            """
            Apply the image adapter of the answer branch
            """

            ans_img_feat, _ = self.ans_adapter(frcn_feat, grid_feat, bbox_feat)

            """
            Process the language features for answers
            """
            self.ans_rnn.flatten_parameters()
            ans_feat = self.ans_embedding(ans_ix)
            ans_feat = self.ans_rnn(ans_feat)
            
            """
            Pass both image and ans features through backbone
            """
            ans_branch_feat = self.ans_backbone(ans_img_feat, ans_feat)

        if self.__C.TRAINING_MODE == 'pretraining_ans':
            """
            Apply the image adapter of the answer branch
            """

            ans_img_feat, _ = self.ans_adapter(frcn_feat, grid_feat, bbox_feat)

            """
            Process the language features for answers
            """
            self.ans_rnn.flatten_parameters()
            ans_feat = self.ans_embedding(ans_ix)
            ans_feat = self.ans_rnn(ans_feat)
            
            """
            Pass both image and ans features through backbone
            """
            ans_branch_feat = self.ans_backbone(ans_img_feat, ans_feat)


        if self.__C.TRAINING_MODE == 'original':
            return self.classifier(ques_branch_feat)

        elif self.__C.TRAINING_MODE == 'pretraining_ans':
            return self.classifier(ans_branch_feat)

        elif self.training:
            # If training we will have to use the answer branch
        
            """
            Add noise to help create a homogeneous space
            """

            # randomly sample a number 'u' between zero and one
            u = torch.rand(1).cuda()

            ques_branch_noise = self.__C.QUES_STDDEV * torch.randn(ques_branch_feat.shape).cuda()
            ans_branch_noise = self.__C.ANS_STDDEV * torch.randn(ans_branch_feat.shape).cuda()
            
            ques_branch_feat += ques_branch_noise
            ans_branch_feat += ans_branch_noise

            fused_feat = torch.add(torch.mul(u, ques_branch_feat), torch.mul(1-u, ans_branch_feat))

            """
            Save the features for visualizations
            """

            # For calculating Fusion Loss in train_engine
            # also normalize the vectors before calculating loss
            z_ques_branch = F.normalize(ques_branch_feat.clone(), p=2, dim=1)
            z_ans_branch = F.normalize(ans_branch_feat.clone(), p=2, dim=1)
            z_fused = F.normalize(fused_feat.clone(), p=2, dim=1)

            if (step < self.num):
                self.z_ques_branch[(step*self.batch_size):((step+1)*self.batch_size)] = ques_branch_feat.clone().detach().cpu().numpy()
                self.z_ans_branch[(step*self.batch_size):((step+1)*self.batch_size)] = ans_branch_feat.clone().detach().cpu().numpy()
                self.z_fused[(step*self.batch_size):((step+1)*self.batch_size)] = fused_feat.clone().detach().cpu().numpy()
            elif (step == self.num):
                np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_proj_' + str(epoch) + '.npy', self.z_proj)
                np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_ans_' + str(epoch) + '.npy', self.z_ans)
                np.save(self.__C.SAVED_PATH + '/' + self.__C.VERSION + '/z_fused_' + str(epoch) + '.npy', self.z_fused)

            elif (step == (self.num + 1)):
                self.z_ques_branch = np.zeros(shape=self.shape)
                self.z_ans_branch = np.zeros(shape=self.shape)
                self.z_fused = np.zeros(shape=self.shape)


            """
            Apply the classifier layer
            """
            ques_branch_feat = self.classifier(ques_branch_feat)
            ans_branch_feat = self.classifier(ans_branch_feat)
            fused_feat = self.classifier(fused_feat)

            return ques_branch_feat, ans_branch_feat, fused_feat, z_ques_branch, z_ans_branch, z_fused

        else:
            return self.classifier(ques_branch_feat)



        
