# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, json, torch, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from openvqa.models.model_loader import ModelLoader
from openvqa.datasets.dataset_loader import EvalLoader
from visualization.wandb_utils import log_visualisations, log_losses

from collections import defaultdict

# Evaluation
@torch.no_grad()
def test_engine(__C, dataset, state_dict=None, validation=False, **kwargs):

    # Load parameters
    if __C.CKPT_PATH is not None:
        print('Warning: you are now using CKPT_PATH args, '
              'CKPT_VERSION and CKPT_EPOCH will not work')

        path = __C.CKPT_PATH
    else:
        path = __C.CKPTS_PATH + \
               '/ckpt_' + __C.CKPT_VERSION + \
               '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

    # val_ckpt_flag = False
    if state_dict is None:
        # val_ckpt_flag = True
        print('Loading ckpt from: {}'.format(path))
        state_dict = torch.load(path)['state_dict']
        print('Finish!')

        if __C.N_GPU > 1:
            state_dict = ckpt_proc(state_dict)

    # Store the prediction list
    # qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_ix_list = []
    pred_list = []

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    if __C.TOY_DATASET:
        ques_ids = dataset.ques_ids

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.eval()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    net.load_state_dict(state_dict)

    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
    )

    # VISUALISATION #
    rand_step = np.random.randint(1, len(dataloader))
    vis_dict = defaultdict(list)
    vis_uids_epoch = []
    num_images_each = 8
    vis_uids_constant = []

    train_image_numbers = [524291000, 393227003, 393268002, 393290000, 131093002, 25010, 262221001, 86001]
    val_image_numbers = [393225003, 262162001, 510657001, 262242007, 393267000, 262161012, 262235002, 262197007]

    if (validation):
        vis_uids_constant = val_image_numbers
    elif ('train' in kwargs and kwargs['train'] == True):
        vis_uids_constant = train_image_numbers

    loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")
    loss_sum = 0

    for step, (batch_dict) in enumerate(dataloader):

        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / __C.EVAL_BATCH_SIZE),
        ), end='          ')


        frcn_feat_iter = batch_dict['frcn_feat_iter'].cuda()
        grid_feat_iter = batch_dict['grid_feat_iter'].cuda()
        bbox_feat_iter = batch_dict['bbox_feat_iter'].cuda()
        ques_ix_iter = batch_dict['ques_ix_iter'].cuda()

        ans_iter = batch_dict['ans_iter'].cuda()
        ques_ids_iter = batch_dict['ques_id']
        
        uids_iter = [ques_ids_iter[i] for i in range(len(ques_ids_iter))]

        pred = net(
            frcn_feat_iter,
            grid_feat_iter,
            bbox_feat_iter,
            ques_ix_iter
        )

        loss_item = [pred['proj_feat'], ans_iter]
        loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
        for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
            if loss_nonlinear in ['flat']:
                loss_item[item_ix] = loss_item[item_ix].view(-1)
            elif loss_nonlinear:
                loss_item[item_ix] = eval('F.' + loss_nonlinear + '(loss_item[item_ix], dim=1)')

        loss = loss_fn(loss_item[0], loss_item[1])
        loss_sum += loss.cpu().data.numpy()

        pred_proj_feat = pred['proj_feat']
        pred_np = pred_proj_feat.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)

        # Save the answer index
        if pred_argmax.shape[0] != __C.EVAL_BATCH_SIZE:
            pred_argmax = np.pad(
                pred_argmax,
                (0, __C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                mode='constant',
                constant_values=-1
            )

        ans_ix_list.append(pred_argmax)
        
        if (step == 0 and len(vis_uids_constant) < num_images_each):
            vis_uids_constant += uids_iter[:num_images_each-len(vis_uids_constant)]

        # Check if the current uids_iter contains any uid from vis_uids_constant
        for idx, uid in enumerate(uids_iter):
            if (uid in vis_uids_constant):
                vis_uids_epoch.append(uid)
                vis_dict['idx'].append(idx)
                all_dict = {**batch_dict, **pred}

                predicted_ans_str = dataset.ix_to_ans[str(pred_argmax[idx])]
                all_dict['predicted_answer'] = predicted_ans_str

                vis_dict['batch_dict'].append(all_dict)

        # Save the whole prediction vector
        if __C.TEST_SAVE_PRED:
            if pred_np.shape[0] != __C.EVAL_BATCH_SIZE:
                pred_np = np.pad(
                    pred_np,
                    ((0, __C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                    mode='constant',
                    constant_values=-1
                )

            pred_list.append(pred_np)

    loss_sum_dict = {}
    loss_sum_dict[__C.LOSS_FUNC] = loss_sum / data_size

    # vis_dict = {
    #    'idx': [idx_1, idx_2, ...],
    #    'batch_dict': [batch_dict_1, batch_dict_2, ...]
    #     idx_i is the idx of the image in batch_dict_i
    # }
    vis_dict['uids_constant'] = vis_uids_constant
    vis_dict['uids_epoch'] = vis_uids_epoch
    
    epoch = kwargs['epoch']

    if ('train' in kwargs and kwargs['train'] == True):
        log_visualisations(__C, vis_dict, mode='train', epoch=kwargs['epoch'], is_mib=False)
        log_losses(loss_sum_dict, mode='train', epoch=epoch)
    elif (validation):
        log_visualisations(__C, vis_dict, mode='val', epoch=kwargs['epoch'], is_mib=False)
        log_losses(loss_sum_dict, mode='val', epoch=epoch)

    print('')
    ans_ix_list = np.array(ans_ix_list).reshape(-1)


    if validation:
        if __C.RUN_MODE not in ['train']:
            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.CKPT_VERSION
        else:
            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.VERSION
    else:
        if __C.CKPT_PATH is not None:
            result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION
        else:
            result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH)


    if __C.CKPT_PATH is not None:
        ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '.pkl'
    else:
        ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH) + '.pkl'


    if __C.RUN_MODE not in ['train']:
        log_file = __C.LOG_PATH + '/log_run_' + __C.CKPT_VERSION + '.txt'
    else:
        log_file = __C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt'

    if __C.TOY_DATASET:
        EvalLoader(__C).eval(dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, ques_ids, validation, **kwargs)
    else:
        EvalLoader(__C).eval(dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, None, validation, **kwargs)


def ckpt_proc(state_dict):
    state_dict_new = {}
    for key in state_dict:
        state_dict_new['module.' + key] = state_dict[key]
        # state_dict.pop(key)

    return state_dict_new
