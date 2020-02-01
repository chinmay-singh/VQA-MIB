from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
#TODO:: Change the class to be imported after implementing backbone
from openvqa.models.positional.positional import MCA_ED
from openvqa.models.positional.positional import Adapter

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch
import math
import numpy as np

#---------------------------------------------------------------
#----------- Model based on postional transformer -------------
#--------------------------------------------------------------

'''
Description of the model:
    Input: Image and the questions
    Method: 
        Preprocessing of Image:
                    Find the bottom up features of the image using FRCNN
        Preprocessing of the questions:
                    Find the Glove+BiLSTM embeddings of the question (max tokens = 14)
        
        step 1: Attention on the objects
                    Find the top-down attention on the obtained image features using the 
                    obtained question features

        step 2: 
                    Find the combined embeddings of all the objects using the positional relations between them
                    Use a transformer to find embeddnigs of each pair of objects
                    Use a suitable method to combine all the embeddings since there can be a variable number of
                    objects in an image
        step 3:
                    Now, use the questions to query over the obtained embeddings to create a vector which can 
                    be taken softmax over to find the right anwer

        Motivation:
                    Think of how humans process a question over an image: First find the relavant objects 
                    and their relavant information, then find the relations between the objects and then 
                    answer the question


'''

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, pretrain_emb_ans, token_size_ans, noise_sigma = 0.1):

        super(Net, self).__init__()

        self.__C = __C

        # Define all the layers that you require in here
        
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)
        
        #TODO:: Change the backbone class after implementing backbone
        self.backbone = MCA_ED(__C)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix ):

        # Pre-process the language features
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))

        # Returns (batch, 14, WORD_EMBED_SIZE)
        lang_feat = self.embedding(ques_ix)

        # Input (batch, 14, input_size_of_lstm = WORD_EMBED_SIZE)
        # output (batch, 14, NUM_DIRECTIONS * HIDDEN_SIZE)
        # h_n (batch, NUM_DIRECTIONS * NUM_LAYERS, HIDDEN_SIZE)
        # c_n (batch, NUM_DIRECTIONS * NUM_LAYERS, HIDDEN_SIZE)
        lang_feat, _ = self.lstm(lang_feat)

        # Now we need to use the img (frcn + bbox feat) to obtain the final vector
        # We need to modify the adapter so as to return bbox features also
        img_feat, img_feat_mask, bbox_feat = self.adapter(frcn_feat, grid_feat, bbox_feat) # (batch, 100, 512), (batch, 1, 1, 100) (batch, 100, 5)

        # Backbone Framework 
        # (batch, 14, 512) (batch, 100, 512)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            bbox_feat
        )

        # Implement the corresponding backbone in positional.py and come back



