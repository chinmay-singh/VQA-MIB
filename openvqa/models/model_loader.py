# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# Modified at FrostLabs
# --------------------------------------------------------

from importlib import import_module


class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL_USE
        model_moudle_path = 'openvqa.models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, **args, **kwargs):
        return self.model_moudle.Net(**args, **kwargs)


class CfgLoader:
    def __init__(self, model_use):

        cfg_moudle_path = 'openvqa.models.' + model_use + '.model_cfgs'
        self.cfg_moudle = import_module(cfg_moudle_path)

    def load(self):
        return self.cfg_moudle.Cfgs()
