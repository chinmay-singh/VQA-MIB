import wandb
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from PIL import Image, ImageDraw
import torchvision
import os

# text should be converted to english before sending here
# images should be loaded to tensor before sending here
# dict (images(img_id/path, bbox), attentions(of both images and text), ans, ques, augmented_answer), mode, epoch

def load_images_from_iid(iids: list, ques: list, ans: list, aug_ans: list, pred_ans: list, mode: str='train'):

    folder_path = join(os.getcwd(), "data/vqa/images", mode + "2014")
    # 000000493618.jpg
    tensors = []
    for idx, iid in enumerate(iids):
        l = len(iid)
        full_name = 'COCO_'+str(mode)+'2014_'+'0'*(12-l) + iid
        fpath = join(folder_path, full_name + '.jpg')
        assert os.path.isfile(fpath), f"File does not exist: {fpath}"
        image = Image.open(fpath)
        if (image.mode != 'RGB'):
            image = image.convert('RGB')

        ####################################################################
        ###################### Remove size hardcoding ######################
        ####################################################################
        image = image.resize((228, 228))

        d = ImageDraw.Draw(image)
        d.text((10,10), ques[idx].replace("_", " "), fill=(0,0,0))
        d.text((10,20), ans[idx].replace("_", " "), fill=(0,0,0))
        d.text((10,30), aug_ans[idx].replace("_", " "), fill=(0,0,0))
        d.text((10,40), pred_ans[idx].replace("_", " "), fill=(0,0,0))
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        tensors.append(image_tensor)

    tensors = torch.stack(tensors, dim=0)

    return tensors

def log_losses(losses: dict, mode: str, epoch: int):
    """
    Log the losses

    Args:
        losses (dict): all the losses should be of type float
        mode (str): 'train' or 'val'
        epoch (int): epoch number
    """

    for k, v in losses.items():
        wandb.log({f"{mode}/{k}": v}, step=epoch)


def log_metrics(metrics: dict, mode: str, epoch: int):
    """
    Log the metrics

    Args:
        metrics (dict): all the metrics should be of type float
        mode (str): 'train' or 'val'
        epoch (int): epoch number
    """

    for k, v in metrics.items():
        wandb.log({f"{mode}/{k}": v}, step=epoch)

def get_bbox_data(bboxes: torch.Tensor, att: torch.Tensor):
    # bbox = Tensor(num_objects, 5)
    # bbox 5 = [tl.x, tl.y, br.x, br.y, area]

    # att = Tensor(num_objects, 1)

    # bboxes = [tl.x, tl.y, br.x, br.y, area=(tl.x-br.x)*(tl.y-br.y)]

    # for removing the zeroed objects (padded numbers as objects)
    bboxes_mask = bboxes[:,-1] != 0
    bboxes = bboxes[bboxes_mask]
    att = att[bboxes_mask]

    # for selecting the top k objects (according to attention)
    num_objects = att.shape[0]
    k_objects = min(20, num_objects)
    _, topk_idx = torch.topk(att, dim=0, k=k_objects)
    topk_idx = topk_idx[:,0]
    bboxes = bboxes[topk_idx]
    att = att[topk_idx]

    bbox_data = []

    att = torch.squeeze(att) # (num_objects, 1)

    for idx, bb in enumerate(bboxes):
        
        bb_dict = {
                "position": {
                    "minX": bb[0].item(),
                    "maxX": bb[2].item(),
                    "minY": bb[1].item(),
                    "maxY": bb[3].item()
                },
                "domain": "percentage",
                "scores" : {
                    "att": att[idx].item()
                },
                "box_caption" : "%s: %.4f" % (idx, att[idx].item()),
                "class_id" : idx
        }
        bbox_data.append(bb_dict)
    
    return bbox_data, bboxes_mask, topk_idx, k_objects

def plot_attention_matrix(ca_list: list, rel_idx: int, abs_idx: int, mode: str, epoch: int, block_number: int, head_number: int, type_att='ca', x_labels='auto', y_labels='auto'):
    # block number in the plot_attention_matrix function refers to the index of the sga module (there are multiple sga modules)
    # head numbers in the plot_attention_matrix function refers to the head (of multi-head attention) within a sga block 
    # "ca": list(Tensor([batch, n_heads, num_objects, num_words]))

    ca_block = ca_list[block_number].detach().cpu()
    ca = ca_block[rel_idx,head_number,:,:] # shape is (num_object, num_words)
    ax = sns.heatmap(ca, xticklabels=x_labels, yticklabels=y_labels, cmap=sns.cm.rocket_r)
    plt.sca(ax)
    wandb.log({f'{mode}/{abs_idx}/{type_att}/{block_number}_{head_number}': wandb.Image(plt)}, step=epoch)
    plt.close()

def plot_image_attentions(image: torch.Tensor, bbox_data: dict, mode: str, epoch: int, abs_idx: int, image_number: int = 0):
    img = wandb.Image(image, boxes={
            "predictions": {
                "box_data": bbox_data,
            }
    })
    if (image_number == 0):
        wandb.log({f'{mode}/{abs_idx}/image': img}, step=epoch)
    else:
        wandb.log({f'{mode}/{abs_idx}/image_{image_number}': img}, step=epoch)

def log_mcan_att(ret: dict, mode: str, epoch: int, rel_idx: int, abs_idx: int):
    """
    ret = {
        "sa": list(Tensor([batch, n_heads, num_objects, num_objects])),
        "ca": list(Tensor([batch, n_heads, num_objects, num_words])),
        "att_flat": list(Tensor([batch, num_objects, 1])), The list here is of size 1 - stupid decision
        "images": Tensor(1, num_channels, h, w),
        "bbox": Tensor(1, num_objects, 5),
        "ques_ix_iter": numpy.ndarray(1, max_num_words), the nd.array contains 0 at places where there is no word,
        "question": [word_1, word_2, ...],
        "text_sa": list(Tensor([batch, n_heads, num_words, num_words])),
        "tex_att_flat": "list(Tensor([batch, num_words, 1]))"

    }
    rel_idx: the index at the batch dimension to be used (where batch is present)
    """

    image = ret['images'][0].detach().cpu() # (num_channels, h, w)
    bboxes = ret['bbox'][0].detach().cpu() # (num_objects, 5)
    att_flat = ret['att_flat'][0][rel_idx].detach().cpu() # (num_objects, 1)
    bbox_data, bboxes_mask, topk_idx, k_objects = get_bbox_data(bboxes, att_flat)
    
    plot_image_attentions(image, bbox_data, mode, epoch, abs_idx)
    
    # Select all the non-zero words
    lang_zero_mask = ret['ques_ix_iter'][0] != 0

    # we have to apply two masks for objects (for selecting available objects and for top 20 objects)
    # we have to apply one mask for language (for selecting available words)
    new_ret = {
        "sa": [],
        "ca": [],
        "text_sa": []
    }
    for idx in range(len(ret['sa'])):
        _sa = ret['sa'][idx].detach().clone()
        _sa = _sa[:,:,bboxes_mask,:]
        _sa = _sa[:,:,:,bboxes_mask]
        _sa = _sa[:,:,topk_idx,:]
        _sa = _sa[:,:,:,topk_idx]
        new_ret['sa'].append(_sa)

        _text_sa = ret['text_sa'][idx].detach().clone()
        _text_sa = _text_sa[:,:,:,lang_zero_mask]
        _text_sa = _text_sa[:,:,lang_zero_mask,:]
        new_ret['text_sa'].append(_text_sa)

    for idx in range(len(ret['ca'])):
        _ca = ret['ca'][idx].detach().clone()
        _ca = _ca[:,:,bboxes_mask,:]
        _ca = _ca[:,:,topk_idx,:]
        _ca = _ca[:,:,:,lang_zero_mask]
        new_ret['ca'].append(_ca)

    # sa
    num_heads = ret['ca'][0].shape[1]
    num_words = ret['ca'][0].shape[3]

    # block number in the plot_attention_matrix function refers to the index of the sga module (there are multiple sga modules)
    # head numbers in the plot_attention_matrix function refers to the head (of multi-head attention) within a sga block 

    language_axis = ret['question'][0].split("_")
    object_axis = list(range(k_objects))

    # list of (block number, head number) tuples
    block_head_number = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
    for (block_number, head_number) in block_head_number:

        # Plotting (text, image) cross attention
        plot_attention_matrix(ca_list=new_ret['ca'], rel_idx=rel_idx, abs_idx=abs_idx, mode=mode, epoch=epoch, block_number=block_number, head_number=head_number, x_labels=language_axis, y_labels=object_axis)

        # Plotting image self attention
        plot_attention_matrix(ca_list=new_ret['sa'], rel_idx=rel_idx, abs_idx=abs_idx, mode=mode, epoch=epoch, block_number=block_number, head_number=head_number, type_att='img_sa' ,x_labels=object_axis, y_labels=object_axis)
        
        # Plotting text self attention
        plot_attention_matrix(ca_list=new_ret['text_sa'], rel_idx=rel_idx, abs_idx=abs_idx, mode=mode, epoch=epoch, block_number=block_number, head_number=head_number, type_att='text_sa' ,x_labels=language_axis, y_labels=language_axis)
        
def log_mf_att(ret: dict, mode: str, epoch: int, rel_idx: int, abs_idx: int):
    """
    ret = {
        "iatt_maps": torch.Tensor(batch, padded_num_objects, num_images_glimpses),
        "images": torch.Tensor(1, c, h, w),
        "bbox": torch.Tensor(1, padded_num_objects, 5),
        "ques_ix_iter": torch.Tensor(1, padded_num_words),
        "question": ['word1_word2_words3.._lastWord'],
        "text_qatt": torch.Tensor(batch, padded_num_words, num_text_glimpses)
    }
    """
    image = ret['images'][0].detach().cpu() # (c, h, w)
    bboxes = ret['bbox'][0].detach().cpu() # (padded_num_objects, 5)
    att = ret['iatt_maps'][rel_idx].detach().cpu() # (padded_num_objects, num_images_glimpses)
    text_att = ret['text_qatt'] # (batch, padded_num_words, num_text_glimpses)

    bbox_data, _, _, _ = get_bbox_data(bboxes, att[:, 0].unsqueeze(1))
    plot_image_attentions(image, bbox_data, mode, epoch, abs_idx, image_number=0)

    bbox_data_2, _, _, _ = get_bbox_data(bboxes, att[:, 1].unsqueeze(1))
    plot_image_attentions(image, bbox_data_2, mode, epoch, abs_idx, image_number=1)

    lang_zero_mask = ret['ques_ix_iter'][0] != 0
    text_att = text_att[:, lang_zero_mask, :]

    language_axis = ret['question'][0].split("_")
    object_axis = [1]

    block_head_number = [(0, 0), (0, -1)]
    for (block_number, head_number) in block_head_number:

        # plot_attention_matrix requires ca_list as: list(Tensor([batch, n_heads, num_objects, num_words]))
        # text_att - (batch, padded_num_words, num_text_glimpses)
        tatt = text_att.detach().clone().cpu()
        tatt = tatt.permute(0, 2, 1) # (batch, num_words, num_glimpses) -> (batch, num_glimpses, num_words)
        tatt = tatt.unsqueeze(2) # (batch, num_glimpses, num_words) -> (batch, num_glimpses, 1, num_words)
        tatt = [tatt] # [(batch, num_glimpses, 1, num_words)]
        plot_attention_matrix(ca_list=tatt, rel_idx=rel_idx, abs_idx=abs_idx, mode=mode, epoch=epoch, block_number=block_number, head_number=head_number, x_labels=language_axis, y_labels=object_axis)

def log_visualisations(__C, ret_dict: dict, mode: str, epoch: int, is_mib: bool):
    """
    vis_dict = {
        'uids_epoch': [4567, 1234, 5678 ,7654] ,
        'uids_constant': [1234, 5678] ,
        'idx': [idx_0, idx_1, idx_2, idx_3],
        'predicted_ans': [ans_1, ans_2, ans_3, ..],
        'batch_dict': [batch_dict_1, batch_dict_2, ...]
        idx_i is the idx of the image in batch_dict_i
    }
    """
    abs_idx = 0
    for uid in ret_dict['uids_constant']:

        idx_in_epoch = ret_dict['uids_epoch'].index(uid)
        rel_idx = ret_dict['idx'][idx_in_epoch]

        _log_visualisations(__C, ret_dict['batch_dict'][idx_in_epoch], rel_idx, abs_idx, mode, epoch, is_mib)
        abs_idx += 1

    for idx_in_epoch, uid in enumerate(ret_dict['uids_epoch']):

        if (uid in ret_dict['uids_constant']):
            continue

        rel_idx = ret_dict['idx'][idx_in_epoch]
        _log_visualisations(__C, ret_dict['batch_dict'][idx_in_epoch], rel_idx, abs_idx, mode, epoch, is_mib)
        abs_idx += 1

def _log_visualisations(__C, ret_dict: dict, rel_idx: int, abs_idx: int, mode: str, epoch: int, is_mib: bool):
    """
        Returned keys by dataloader: ['ques_ix_iter', 'ans_iter', 'iid', 'question', 'augmented_answer', 
        'answer', 'question_type', 'frcn_feat_iter', 'grid_feat_iter', 'bbox_feat_iter']

        Returned keys by net: ["proj_feat", "img", "text"]
    """
    iids = ret_dict['iid'][rel_idx:rel_idx+1]
    ans = ret_dict['answer'][rel_idx:rel_idx+1]
    aug_ans = ret_dict['augmented_answer'][rel_idx:rel_idx+1]
    ques = ret_dict['question'][rel_idx:rel_idx+1]
    ques_ix_iter = ret_dict['ques_ix_iter'][rel_idx:rel_idx+1]
    predicted_answer = [ret_dict['predicted_answer']]

    # Load text keys when ret_dict hasn't been modified
    text_dict = {}
    for key, value in ret_dict['text'].items():
        text_dict['text_' + str(key)] = value

    images = load_images_from_iid(iids, ques, ans, aug_ans, predicted_answer, mode)
    ret_dict['images'] = images
    ret_dict['bbox'] = ret_dict['bbox_feat_iter'][rel_idx:rel_idx+1]

    if (is_mib):

        ### image attn ###

            ### ques branch
            img_dict = {}
            img_dict = ret_dict['ques']['img']
            img_dict['images'] = ret_dict['images']
            img_dict['bbox'] = ret_dict['bbox']
            log_image_att(img_dict, mode, epoch, rel_idx, abs_idx)


            ### ans branch
            img_dict = {}
            img_dict = ret_dict['ans']['img']
            img_dict['images'] = ret_dict['images']
            img_dict['bbox'] = ret_dict['bbox']
            log_image_att(img_dict, mode, epoch, rel_idx, abs_idx)
    else:

        ### ques branch
        both_dict = {}
        both_dict = ret_dict['img']
        both_dict['images'] = ret_dict['images'].detach().cpu()
        both_dict['bbox'] = ret_dict['bbox'].detach().cpu()
        both_dict['ques_ix_iter'] = ques_ix_iter.detach().cpu()
        both_dict['question'] = ques

        both_dict = {**both_dict, **text_dict}
        if __C.MODEL == "mcan_small" or __C.MODEL == "mcan_large":
            log_mcan_att(both_dict, mode, epoch, rel_idx, abs_idx)
        elif __C.MODEL == "mfb" or __C.MODEL =="mfh":
            log_mf_att(both_dict, mode, epoch, rel_idx, abs_idx)

