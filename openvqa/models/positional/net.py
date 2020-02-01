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
