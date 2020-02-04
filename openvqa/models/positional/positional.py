from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x

# ------------------------------------------------
# ---- Positional Transformer ----
# ------------------------------------------------

class PT(nn.Module):
    def __init__(self, __C):
        super(PT, self).__init__()

        # initialize the MLP for expanding the 
        self.mlp = MLP(
            in_size= 5,
            mid_size=512,
            out_size=1024,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        # initialize a transformer

        self.trans = nn.Transformer(d_model = 1024)


    def calculate_relative_postitions(self,z1, z2):
        x_center_1 = (z1[0] + z1[2]) / 2
        y_center_1 = (z1[1] + z1[3]) / 2
        
        x_center_2 = (z2[0] + z2[2]) / 2
        y_center_2 = (z2[1] + z2[3]) / 2


        l_r = torch.sign(x_center_1 - x_center_2)
        
        u_d = torch.sign(y_center_1 - y_center_2)


        dist = torch.dist(torch.tensor([x_center_1,y_center_1]),torch.tensor([x_center_2,y_center_2]))


        return(torch.Tensor([l_r ,u_d , dist , z1[4] ,z2[4]])) # rather than passing all the things, it will be better if we pass relevant things only

    def make_src_trgt(self, x, z):

        '''
        input :
            x: input attended object features (batch, 100, 512)
            z: input bbox features (batch, 100, 13)
        '''
        
        pairwise_pos_vectors = []
        pairwise_feature_vectors = []

        for i in range(0,x.shape[0]): # for every example in the batch
            temp_pos = []
            temp_feat = []
            for j in range(0,x.shape[1]): # for every object in each example
                for k in range(j+1, x.shape[1]): # for every other object in the example
                    temp_pos.append(self.calculate_relative_postitions(z[i][j],z[i][k]))  # SELF.CALCULATE_RELATIVE_POSITOIN GIVES 1*13
                    temp_feat.append((x[i][j]+x[i][k])/2) #APPENDED 1*512

            # after these nested loops, for each example in batch, temp_pos is a list containing 100C2 5 dimensional vectors

            pairwise_pos_vectors.append(torch.stack(temp_pos))              #BATCH HENCE TEMP_POS IS 100C2 *5, here it is a list
            pairwise_feature_vectors.append(torch.stack(temp_feat))         #BATCH HENCE TEMP_feat IS 100C2 *512, here it is a list
        
        pairwise_pos_vectors = torch.stack(pairwise_pos_vectors).cuda()            #bATCH SIZE* 100c2 *5, here it is a tensor    
        pairwise_feature_vectors = torch.stack(pairwise_feature_vectors).cuda()      #BATCH SIZE * 100C2 *512, here it is a tensor
        
        return pairwise_feature_vectors , pairwise_pos_vectors               # (batch, 100C2, 512), (batch, 100C2, 5)

    def forward(self, x, z):
        
        # take a pairwise combination of each image_feat vector
        # and do the corresponding calculations for it

        assert x.shape[0] == z.shape[0] and x.shape[1] == z.shape[1], "Shapes of object features and bbox featuers do not match"

        trgt, src = self.make_src_trgt(x,z) # the positional vectors are treated as source for the transformer

        #change dimension of source with mlp
        
        src = self.mlp(src) # now src is (batch, 100C2, 512) [pytorch takes the last dimension itself]

        assert src.shape[2] == trgt.shape[2], "src shape is {} and trgt shape is {}".format(src.shape, trgt.shape)

        print("Assert Success")

        #apply transformer

        temp_batch = []
        for i in range(0, src.shape[0]): # for every example in the batch
            temp_example = []
            for j in range(0,src.shape[1]): # for every pair of objects
                temp_example.append(self.trans(src[i][j].unsqueeze(0).unsqueeze(0), trgt[i][j].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0))

            temp_batch.append(torch.stack(temp_example))

        return torch.stack(temp_batch)                                      # (batch, 100C2, 512)

# ------------------------------------------------
# ---- Attention Whore ----
# ------------------------------------------------

class AW(nn.Module):
    def __init__(self, __C):
        super(AW, self).__init__()
        
        # initialize an object of modular co attention encoder decoder
        self.mcan = MCA_ED(__C)

        # initialize an object of Positional Transformer
        self.pt = PT(__C)
        
        # Positional Transformer

        self.transformer = nn.Transformer()

        # Rest can be made later on
    def forward(self, y, x, y_mask, x_mask, z):
        '''
        inputs:
            y: question embeddings (batch, 14, 1 * 512) (if bidirectional, it will be 2 * 512)
            x: frcn_feat (batch, 100, 512)
            y_mask: lang_feat mask
            x_mask: img_feat_mask (batch, 1, 1, 100)
            z: bbox features (batch, 100, 5)
        '''

        # Find the embeddings of each object and lang features after passing through
        # the modular co attention encoder and decoder
        y, x = self.mcan(y, x, y_mask, x_mask)

        # Now use the positional transformer to find whatever is needed 
        
        p = self.pt(x , z)       # (batch, 100C2, 512)

        return y, p
