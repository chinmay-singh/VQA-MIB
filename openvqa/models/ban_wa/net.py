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

        self.adapter = Adapter(__C)
        self.backbone = BAN(__C)


        # Classification layers
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE, __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifier = nn.Sequential(*layers)
        
        # create the noise vector std
        self.noise_sigma = noise_sigma


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, ans_ix):

        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.rnn(lang_feat)
        
        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        lang_feat = self.backbone(
            lang_feat,
            img_feat
        )

        # sum the lang+img features along dimesion 1
        lang_feat = lang_feat.sum(1)
        
        # create a noise vector
        noise_vec = self.noise_sigma*torch.randn(lang_feat.shape).cuda()

        # add the noise to lang+img features
        lang_feat += noise_vec

        # Classification layers
        proj_feat = self.classifier(lang_feat)
        
        # ans features
        ans_feat = self.ans_embedding(ans_ix)
        ans_feat, _ = self.ans_rnn(ans_feat)
        ans_feat = ans_feat.sum(1)

        # add the same noise to ans_feat but only at training time
        if not self.eval_flag:
            assert ans_feat.shape == lang_feat.shape, "ans_feat: {} and lang_feat: {} shapes do not match".format(ans_feat.shape, lang_feat.shape)
            ans_feat += noise_vec
        
        # classification layer
        ans_proj_feat = self.classifier(ans_feat)

        
        # randomly sample a number 'u' between zero and one
        u = torch.rand(1).cuda() 

        # now we can fuse the vector
        if not self.eval_flag:
            fused_feat = torch.add(torch.mul(u, lang_feat), torch.mul(1-u, ans_feat))
            fused_proj_feat = self.classifier(fused_feat)
        else:
            fused_feat = proj_feat
            fused_proj_feat = self.classifier(fused_feat)

        return proj_feat, ans_proj_feat, fused_feat
