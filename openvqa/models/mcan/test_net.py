from openvqa.models.mcan.net import Net
from openvqa.models.mcan.net_mib import Net as NetMib
from openvqa.models.model_loader import CfgLoader
import yaml
import torch
from collections import defaultdict
# from torchsummary import summary

def print_requires_grad(net: torch.nn.Module):
    print("############# Printing network grads ###################")
    for name, p in net.named_parameters():
        print (name, p.requires_grad)

def load_config():
    cfg_file = "configs/vqa/mcan_small.yml"
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.safe_load(f)
        
    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    __C.DATASET = 'vqa'

    __C.USE_GLOVE = False
    __C.RUN_MODE = 'train'

    args_dict = {**yaml_dict}
    __C.add_args(args_dict)
    __C.proc()

    return __C

def printer(v, tab=1):

    if (isinstance(v, int) or isinstance(v, str)):
        print('\t'*tab, v)
    elif (isinstance(v, torch.Tensor)):
        print('\t'*tab, v.shape)
    elif (isinstance(v, dict) or isinstance(v, defaultdict)):
        for k_, v_ in v.items():
            print('\t'*tab, k_)
            printer(v_, tab+1)
    elif (isinstance(v, list)):
        for v_ in v:
            printer(v_, tab)

def print_count_requires_grads(net):
    frozen = 0
    not_frozen = 0
    total = 0
    for n, p in net.named_parameters():
        if (p.requires_grad == True):
            not_frozen += 1
        else:
            frozen += 1
    total = frozen + not_frozen
    print(f"{frozen}/{total} parameters frozen")
    print(f"{not_frozen}/{total} parameters training")

def test_net():

    batch_size = 32
    __C = load_config()
    n = Net(__C, pretrained_emb=None, token_size=10000, answer_size=3129)

    print("############### REQUIRES GRAD COUNT #################")
    print_count_requires_grads(n)

    frcn_feat_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE']
    frcn_feat = torch.randn(batch_size, *frcn_feat_size)

    grid_feat = []

    bbox_feat_size = __C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE']
    bbox_feat = torch.randn(batch_size, *bbox_feat_size)

    text_ix = torch.randint(low=0, high=14, size=(batch_size,14)).squeeze()

    out = n.forward(frcn_feat=frcn_feat, grid_feat=None, bbox_feat=bbox_feat, text_ix=text_ix)
    print("############### NET KEYS #################")
    print(n.return_dict())

    print("############### FORWARD PASS #################")
    for k, v in out.items():
        print(k)
        printer(v)
    
    print("############### PRINTING NETWORK #################")
    print(n)


def test_mib_net():

    batch_size = 32
    __C = load_config()
    n = NetMib(__C, pretrained_emb_ques=None, pretrained_emb_ans=None ,token_size=10000, answer_size=3129)
    print("############### REQUIRES GRAD COUNT #################")
    print_count_requires_grads(n)

    frcn_feat_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE']
    frcn_feat = torch.randn(batch_size, *frcn_feat_size)

    grid_feat = []

    bbox_feat_size = __C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE']
    bbox_feat = torch.randn(batch_size, *bbox_feat_size)

    ans_len = 10
    ques_len = 14
    ques_ix = torch.randint(low=0, high=ques_len, size=(batch_size,ques_len)).squeeze()
    ans_ix = torch.randint(low=0, high=ans_len, size=(batch_size,ans_len)).squeeze()

    out = n.forward(frcn_feat=frcn_feat, grid_feat=None, bbox_feat=bbox_feat, ques_ix=ques_ix, ans_ix=ans_ix)
    print("############### NET KEYS #################")
    print(n.return_dict())

    print("############### FORWARD PASS #################")
    for k, v in out.items():
        print(k)
        printer(v)
    
    print("############### PRINTING NETWORK #################")
    print(n)
