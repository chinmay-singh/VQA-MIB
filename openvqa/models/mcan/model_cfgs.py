# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        # --------------- #
        # ---- NOISE ---- #
        # --------------- #

        self.PROJ_STDDEV = 0.1
        self.ANS_STDDEV = 0.01

        # --------------
        # ---- LSTM ----
        # --------------

        # Properties for the LSTM that encodes the question
        self.LSTM_LAYERS = 1

        # 1 for unidirectional LSTM, 2 for bidirectional LSTM
        self.LSTM_NUM_DIRECTIONS = 1

        # -------------------------
        # ---- ENCODER-DECODER ----
        # -------------------------
        
        # Number of encoders and decoders each
        self.LAYER = 6

        # Number of heads for MultiHeaded Attention
        self.MULTI_HEAD = 8

        # LSTM Hidden state size, 
        self.HIDDEN_SIZE = 512

        # Intermediate size in Feed Forward network (see class FFN in mca.py): HIDDEN_SIZE -> FF_SIZE -> HIDDEN_SIZE
        self.FF_SIZE = 2048

        # Dropout rate 
        self.DROPOUT_R = 0.1

        # NUM_DIRECTIONS * HIDDEN_SIZE -> MLP_SIZE -> FLAT_GLIMPSES
        self.FLAT_MLP_SIZE = 512

        # Attention glimpses while flattening the features
        self.FLAT_GLIMPSES = 1

        # flat out size after flattening the features
        self.FLAT_OUT_SIZE = 1024

        # -----------------------------------
        # ---- USE BOUNDING BOX FEATURES ----
        # -----------------------------------

        # Use Bounding Box features for images
        self.USE_BBOX_FEAT = False

        # linear layer from 5 -> BBOXFEAT_EMB_SIZE
        self.BBOXFEAT_EMB_SIZE = 2048

        # --------------------
        # ---- IRRELEVANT ----
        # --------------------

        # not for VQA dataset 
        self.USE_AUX_FEAT = False

