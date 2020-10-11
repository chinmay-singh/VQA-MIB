# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# Modified at FrostLabs
# --------------------------------------------------------

import os, torch, datetime, shutil, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import wandb
from openvqa.models.model_loader import ModelLoader
from openvqa.utils.optim import get_optim, adjust_lr
from utils1.test_engine import test_engine, ckpt_proc
from vis import plotter, vis_func
from multiprocessing import Pool
import multiprocessing
import sys

def train_engine(__C, dataset, dataset_eval=None):

    print("Model being used is {}".format(__C.MODEL_USE))

    """
    Preparing for initializing the net
    """

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    pretrained_emb_ans = dataset.pretrained_emb_ans
    token_size_ans = dataset.token_size_ans

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size,
        pretrained_emb_ans,
        token_size_ans,
        training=True
    )

    net.cuda()
    net.train()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    # Define Loss Function
    loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")


    # creating a folder for saving the numpy visualization arrays
    if (__C.WITH_ANSWER and ((__C.VERSION) not in os.listdir(__C.SAVED_PATH))):
        os.mkdir(__C.SAVED_PATH + '/' + __C.VERSION)

    if __C.TRAINING_MODE in ['pretrained_ans']:
        print("Using the pretrained model to initialize the answer branch")

        path = __C.CKPTS_PATH + \
               '/ckpt_' + __C.CKPT_VERSION + \
               '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'
        
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

        pretrained_state_dict = torch.load(path)['state_dict']
        if __C.N_GPU > 1:
            pretrained_state_dict = ckpt_proc(pretrained_state_dict)

        net_state_dict = net.state_dict()

        print("updating keys in net_state_dict form pretrained state dict")
        net_state_dict.update(pretrained_state_dict)

        print("loading this state dict in the net")



    # Load checkpoint if resume training
    if __C.RESUME:
        print(' ========== Resume training')

        if __C.CKPT_PATH is not None:
            print('Warning: Now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')
            path = __C.CKPT_PATH
        else:
            path = __C.CKPTS_PATH + \
                   '/ckpt_' + __C.CKPT_VERSION + \
                   '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

        # Load the network parameters
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

        if __C.N_GPU > 1:
            net.load_state_dict(ckpt_proc(ckpt['state_dict']))
        else:
            net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

        # Load the optimizer paramters
        optim = get_optim(__C, net, data_size, ckpt['lr_base'])
        optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(ckpt['optimizer'])
        
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

    else:
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            #shutil.rmtree(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

        optim = get_optim(__C, net, data_size)
        start_epoch = 0

    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))

    
    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM,
        drop_last=True
    )

    logfile = open(
        __C.LOG_PATH +
        '/log_run_' + __C.VERSION + '.txt',
        'a+'
    )
    logfile.write(str(__C))
    logfile.close()

    # For dry runs
    # os.environ['WANDB_MODE'] = 'dryrun' 

    # initializing the wandb project
    wandb.init(project=__C.PROJECT_NAME, name=__C.VERSION, config=__C)

    # obtain histogram of each gradients in network as it trains
    wandb.watch(net, log="all")

    # Training script
    for epoch in range(start_epoch, __C.MAX_EPOCH):

        # Save log to file
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            '=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()

        # Learning Rate Decay
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)

        # Externally shuffle data list
        # if __C.SHUFFLE_MODE == 'external':
        #     dataset.shuffle_list(dataset.ans_list)

        time_start = time.time()
        # Iteration
        for step, (
                frcn_feat_iter,
                grid_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,

                #Edits
                ans_ix_iter,
                #End of Edits

                ans_iter,
                ques_type

        ) in enumerate(dataloader):

            optim.zero_grad()

            frcn_feat_iter = frcn_feat_iter.cuda()
            grid_feat_iter = grid_feat_iter.cuda()
            bbox_feat_iter = bbox_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            #Edits
            ans_ix_iter = ans_ix_iter.cuda()
            #End of Edits
            ans_iter = ans_iter.cuda()

            loss_tmp = 0

            loss_img_ques_tmp = 0
            loss_ans_tmp = 0
            loss_interp_tmp = 0
            loss_fusion_tmp = 0

            for accu_step in range(__C.GRAD_ACCU_STEPS):
                loss_tmp = 0
                loss_img_ques_tmp = 0
                loss_ans_tmp = 0
                loss_interp_tmp = 0
                loss_fusion_tmp = 0

                sub_frcn_feat_iter = \
                    frcn_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_grid_feat_iter = \
                    grid_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_bbox_feat_iter = \
                    bbox_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ques_ix_iter = \
                    ques_ix_iter[accu_step * __C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * __C.SUB_BATCH_SIZE]
                #Edits
                sub_ans_ix_iter = \
                    ans_ix_iter[accu_step * __C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * __C.SUB_BATCH_SIZE]
                #End of Edits

                sub_ans_iter = \
                    ans_iter[accu_step * __C.SUB_BATCH_SIZE:
                             (accu_step + 1) * __C.SUB_BATCH_SIZE]

                
                if __C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans']:
                    pred, pred_ans_branch, pred_fused, z_img_ques, z_ans, z_fused = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter,
                        sub_ans_ix_iter,
                        step,
                        epoch
                    )
                else:
                     pred = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter,
                        sub_ans_ix_iter,
                        step,
                        epoch
                    )
                   
                loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                
                loss_item_pred = [pred, sub_ans_iter]
                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat']:
                        loss_item_img_ques[item_ix] = loss_item_img_ques[item_ix].view(-1)
                    elif loss_nonlinear:
                        loss_item_img_ques[item_ix] = eval('F.' + loss_nonlinear + '(loss_item_img_ques[item_ix], dim=1)')


                # only calculate the ans and interp loss in case of WITH_ANSWER
                if (__C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans']):
                    loss_item_ans = [pred_ans_branch, sub_ans_iter]
                    loss_item_interp = [pred_fused, sub_ans_iter]
                
                    for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                        if loss_nonlinear in ['flat']:
                            loss_item_ans[item_ix] = loss_item_ans[item_ix].view(-1)
                        elif loss_nonlinear:
                            loss_item_ans[item_ix] = eval('F.' + loss_nonlinear + '(loss_item_ans[item_ix], dim=1)')

                    for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                        if loss_nonlinear in ['flat']:
                            loss_item_interp[item_ix] = loss_item_interp[item_ix].view(-1)
                        elif loss_nonlinear:
                            loss_item_interp[item_ix] = eval('F.' + loss_nonlinear + '(loss_item_interp[item_ix], dim=1)')


                # Now we create all the four losses and then add them
                loss = 0
                loss_pred = loss_fn(loss_item_pred[0], loss_item_pred[1])
                loss += loss_pred
                
                if __C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans']:
                    # loss for the prediction from the answer
                    loss_ans = loss_fn(loss_item_ans[0], loss_item_ans[1])
                
                    # Loss for the prediction from the fused vector
                    loss_interp = loss_fn(loss_item_interp[0], loss_item_interp[1])
                    
                    # we also need to multiply this fused loss by a hyperparameter alpha
                    loss_interp *= __C.ALPHA
                    loss += loss_ans + loss_interp

                    if (__C.WITH_FUSION_LOSS):

                        # Now calculate the fusion loss
                        # 1.Higher loss for higher distance between vectors predicted
                        # by different models for same example
                        dist_calc = (z_img_ques - z_ans).pow(2).sum(1).sqrt()
                        loss_fusion = dist_calc.mean()
                        loss_fusion -= torch.pdist(z_img_ques, 2).mean() 
                        loss_fusion -= torch.pdist(z_ans, 2).mean() 

                        # Multiply the loss fusion with hyperparameter beta
                        loss_fusion *= __C.BETA

                        loss += loss_fusion

                
                loss /= __C.GRAD_ACCU_STEPS
                loss.backward()

                loss_tmp += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS

                # calculating temp loss of each type
                if __C.TRAINING_MODE in ['simultaneous_qa', 'pretrained_ans']:
                    loss_pred_tmp += loss_pred.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                    loss_ans_tmp += loss_ans.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                    loss_interp_tmp += loss_interp.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                    if (__C.WITH_FUSION_LOSS):
                        loss_fusion_tmp += loss_fusion.cpu().data.numpy() * __C.GRAD_ACCU_STEPS


            if __C.VERBOSE:
                if dataset_eval is not None:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
                else:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

                print("\r[Version %s][Epoch %2d][Step %4d/%4d] Loss: %.4f [main_pred: %.4f,ans: %.4f,interp: %.4f,fusion: %.4f]" % (
                    __C.VERSION,
                    epoch + 1,
                    step,
                    int(data_size / __C.BATCH_SIZE),
                    loss_tmp / __C.SUB_BATCH_SIZE,
                    loss_pred_tmp / __C.SUB_BATCH_SIZE,
                    loss_ans_tmp / __C.SUB_BATCH_SIZE,
                    loss_interp_tmp / __C.SUB_BATCH_SIZE,
                    loss_fusion_tmp / __C.SUB_BATCH_SIZE
                ), end = ' ')

            # Gradient norm clipping
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )

            # Save the gradient information
            for name in range(len(named_params)):
                norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                    if named_params[name][1].grad is not None else 0
                grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS
                # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                #       (str(grad_wt),
                #        params[grad_wt][0],
                #        str(norm_v)))

            optim.step()

        time_end = time.time()
        elapse_time = time_end-time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Save checkpoint
        if __C.N_GPU > 1:
            state = {
                'state_dict': net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        else:
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        torch.save(
            state,
            __C.CKPTS_PATH +
            '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) +
            '.pkl'
        )
        
        # Logging
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / data_size) +
            ', Lr: ' + str(optim._rate) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) + 
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )
        logfile.close()

        wandb.log({
            'Loss': float(loss_sum / data_size),
            'Learning Rate': optim._rate,
            'Elapsed time': int(elapse_time) 
            })

        # ---------------------------------------------- #
        # ---- Create visualizations in new processes----#
        # ---------------------------------------------- #
        dic = {}
        dic['version'] = __C.VERSION
        dic['epoch'] = epoch 
        dic['num_samples'] = 1000
        p = Pool(processes= 1)
        p.map_async(vis_func, (dic, ))
        p.close()

        # Eval after every epoch
        epoch_dict = {
                'current_epoch': epoch
                }
        __C.add_args(epoch_dict)
        if dataset_eval is not None:
            test_engine(
                __C,
                dataset_eval,
                state_dict=net.state_dict(),
                validation=True,
                epoch = 0
            )
        p.join()

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
