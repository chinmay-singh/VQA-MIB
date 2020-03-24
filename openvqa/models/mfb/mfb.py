# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.ops.fc import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------


class MFB(nn.Module):
    def __init__(self, __C, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, __C.MFB_K * __C.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, __C.MFB_K * __C.MFB_O)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.pool = nn.AvgPool1d(__C.MFB_K, stride=__C.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.__C.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.__C.MFB_O)      # (N, C, O)
        return z, exp_out


class QAtt(nn.Module):
    def __init__(self, __C):
        super(QAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(
            in_size=__C.LSTM_OUT_SIZE,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.Q_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # (N, T, Q_GLIMPSES)
        qatt_maps = F.softmax(qatt_maps, dim=1)         # (N, T, Q_GLIMPSES)

        qatt_feat_list = []
        for i in range(self.__C.Q_GLIMPSES):
            mask = qatt_maps[:, :, i:i + 1]             # (N, T, 1)
            mask = mask * ques_feat                     # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # (N, LSTM_OUT_SIZE*Q_GLIMPSES)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, __C, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.mfb = MFB(__C, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=__C.MFB_O,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.I_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat, bbox_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)
        print("shape of attention map is,     ",iatt_maps.shape)

        #Inserting code to make the attention map
        if self.__C.USE_NEW_QUESTION == "True":        

            print("plotting attention of objects on image")

            img = cv2.imread('/mnt/sdb/yash/data_olddirectory/vqa/coco/raw/test2015/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12) + '.jpg', 3)
            img1 = np.array(img).astype('float64')
            img1 = np.copy(img1)
            img2 = np.copy(img1)
            
            bbox_normalized = bbox_feat.squeeze()
            
            # att_squeezed = i_att.squeeze()
            att_squeezed = []

            for i in range(self.__C.I_GLIMPSES):
                att_squeezed.append(iatt_maps[:, :, i:i + 1].squeeze())

            for i in range(bbox_normalized.shape[0]):

                '''
                if float(att_squeezed[i]) < 0.09:
                    continue
                '''
                x1_coordinate = bbox_normalized[i][0]*float(img.shape[1])
                y1_coordinate = bbox_normalized[i][1]*float(img.shape[0])

                x4_coordinate = bbox_normalized[i][2]*float(img.shape[1])
                y4_coordinate = bbox_normalized[i][3]*float(img.shape[0])
                

                img1[int(y1_coordinate):int(y4_coordinate), int(x1_coordinate):int(x4_coordinate), :] += (255. * float(att_squeezed[0][i].cpu()))
                img2[int(y1_coordinate):int(y4_coordinate), int(x1_coordinate):int(x4_coordinate), :] += (255. * float(att_squeezed[1][i].cpu()))

                img = cv2.rectangle(img, (x1_coordinate, y1_coordinate), (x4_coordinate, y4_coordinate), (255,0,0), 2)


            #making folder for the models name
            save_dir_attention = 'test_images/attention/'+str(self.__C.MODEL_USE)
            if not os.path.exists(save_dir_attention):
                os.mkdir(save_dir_attention)
            
            save_dir_bounding_box = 'test_images/bounding_box/'+str(self.__C.MODEL_USE)
            if not os.path.exists(save_dir_bounding_box):
                os.mkdir(save_dir_bounding_box)
            


            cv2.imwrite(save_dir_attention+'/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12)+"_glimpse1" + '.jpg', img1)
            cv2.imwrite(save_dir_attention+'/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12)+"_glimpse2" + '.jpg', img2)
            cv2.imwrite(save_dir_bounding_box+'/COCO_test2015_' + str(self.__C.IMAGE_ID).zfill(12) + '.jpg', img)

            #############################################
            ########End of Visualization thingy##########







        iatt_feat_list = []
        for i in range(self.__C.I_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat


class CoAtt(nn.Module):
    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C

        img_feat_size = __C.FEAT_SIZE[__C.DATASET]['FRCN_FEAT_SIZE'][1]
        img_att_feat_size = img_feat_size * __C.I_GLIMPSES
        ques_att_feat_size = __C.LSTM_OUT_SIZE * __C.Q_GLIMPSES

        self.q_att = QAtt(__C)
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)

        if self.__C.HIGH_ORDER:  # MFH
            self.mfh1 = MFB(__C, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(__C, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(__C, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat , bbox_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat, bbox_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        if self.__C.HIGH_ORDER:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # (N, O)

        return z
