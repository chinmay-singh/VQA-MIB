import torch
from wandb_utils import log_visualisations
import wandb

def test_log_vis():

    # [tl.x, tl.y, h, w, area]

    ret_dict = {
        'images': torch.randn(32, 3, 300, 300),
        'bbox': torch.randn(32, 100, 5),
        'img': {
            'ca': [torch.randn(32, 8, 100, 14)] * 6,
            'att_flat':[torch.randn(32, 100, 1)]
        }
    }

    wandb.init(project="openvqa_v2", name='test_vis')
    log_visualisations(ret_dict=ret_dict, mode='train', epoch=0, is_mib=False)

    ret_dict = {
        'images': torch.randn(32, 3, 300, 300),
        'bbox': torch.randn(32, 100, 5),
        'img': {
            'ca': [torch.randn(32, 8, 100, 14)] * 6,
            'att_flat':[torch.randn(32, 100, 1)]
        }
    }

    log_visualisations(ret_dict=ret_dict, mode='train', epoch=1, is_mib=False)

test_log_vis()