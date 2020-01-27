# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.ban.ban import BAN
from openvqa.models.ban.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch

# -------------------------
# ---- Main BAN Model With Answers----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, pretrain_emb_ans, token_size_ans, noise_sigma = 0.1):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.ans_embedding = nn.Embedding(
            num_embeddings=token_size_ans,
            embedding_dim=__C.WORD_EMBED_SIZE
        )


        #Edits
        if pretrain_emb_ans != None:
            self.ans_embedding = nn.Embedding(
                num_embeddings=token_size_ans,
                embedding_dim= __C.WORD_EMBED_SIZE
            )
        #End of Edits

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            
            #Edits
            if pretrain_emb_ans != None:
                self.ans_embedding.weight.data.copy_(torch.from_numpy(pretrain_emb_ans))
            #End of Edits

        self.rnn = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = BAN(__C)


        # Classification layers
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE, __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)



        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)

        #Edits
        if ans_ix != None:                          #Change this code? it has been crudely written assumig that None is being Passed
            lang_feat_ans = self.ans_embedding(ans_ix)
        #End of Edits

        lang_feat, _ = self.rnn(lang_feat)

        #Edits
        lang_feat_ans, _ = self.rnn(lang_feat_ans)
        #End of Ends
        '''
        ques_ix contains the batch of natural language ques
        pretrained_emb contains embeddings of all the words that appear in the questions
        self.embedding is a container in which embeddings of all words are copied from pretrained_emb
        self.embedding(ques_ix) becomes a layer which finds embeddings of all words and might be trainable 
        so, we should rewrite this such that this function gets something like ans_to_ix and we can use self.embedding on it
        what changes I need to do:
        In vqa_loader.py:
            make pretrain_emb_ans using answers vocab from answer_dict.json
            preprocess an answer so as to return a dictionary containing the answer words as keys and their positions as indices call it ans_to_ix
            return ans_to_ix from load_ques_ans fn
        
        In base_dataset.py
            return ans_to_ix from getitem
           
        In train_engine.py:
            when enumerating on a batch returned from dataloader, pass ans_to_ix (batch) to the net
            also save dataset.pretrain_emb_ans as attribute and pass in net
            Later: modify loss function

        In net.py:
            pretrain_emb_ans will come from dataset.pretrain_emb
            now use nn.embedding initialized to pretrain_emb_ans and convert answer batch into embeddings using this nn.embedding
            use self.rnn on the answer batch embedding
            combine (image, ques) vector with (ans) vector using random sampling and whatever it is
        '''

        
        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        lang_feat = self.backbone(
            lang_feat,
            img_feat
        )

        # Classification layers
        proj_feat = self.classifer(lang_feat.sum(1))

        return proj_feat
