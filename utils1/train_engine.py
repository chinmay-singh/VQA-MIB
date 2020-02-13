# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
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
# from knockknock import slack_sender

# webhook_url = "https://hooks.slack.com/services/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"

# @slack_sender(webhook_url=webhook_url, channel="terminator")
def train_engine(__C, dataset, dataset_eval=None):

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    #Edits
    pretrained_emb_ans = dataset.pretrained_emb_ans
    token_size_ans = dataset.token_size_ans #End of Edits

    print("Model being used is {}".format(__C.MODEL_USE))

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size,
        pretrained_emb_ans,
        token_size_ans
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

    # Define multi-thread dataloader
    # if __C.SHUFFLE_MODE in ['external']:
    #     dataloader = Data.DataLoader(
    #         dataset,
    #         batch_size=__C.BATCH_SIZE,
    #         shuffle=False,
    #         num_workers=__C.NUM_WORKERS,
    #         pin_memory=__C.PIN_MEM,
    #         drop_last=True
    #     )
    # else:
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
    # TODO to change the name of project later, once the proper coding starts
    wandb.init(project="openvqa", name=__C.VERSION, config=__C)

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

                ans_iter

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
            for accu_step in range(__C.GRAD_ACCU_STEPS):
                loss_tmp = 0

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

                
                # when making predictions also pass the ans_iter which is a dictionary from which you
                # can extract answers and pass them through decoders

                if (__C.WITH_ANSWER):
                    pred_img_ques, pred_ans, pred_fused, z_img_ques, z_ans, z_fused = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter,
                        sub_ans_ix_iter,
                        step,
                        epoch
                    )
                else:
                     pred_img_ques = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter,
                        sub_ans_ix_iter,
                        step,
                        epoch
                    )
                   
                # we need to change the loss terms accordingly
                # now we need to modify the loss terms for the same
                
                #Edits: creating the loss items for each of the prediction vector

                loss_item_img_ques = [pred_img_ques, sub_ans_iter]

                # only calculate the ans and interp loss in case of WITH_ANSWER
                if (__C.WITH_ANSWER):
                    loss_item_ans = [pred_ans, sub_ans_iter]
                    loss_item_interp = [pred_fused, sub_ans_iter]

                
                loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                
                # applying the same transformation on the all three
                # althought for 'bce' loss the following does nothing
                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat']:
                        loss_item_img_ques[item_ix] = loss_item_img_ques[item_ix].view(-1)
                    elif loss_nonlinear:
                        loss_item_img_ques[item_ix] = eval('F.' + loss_nonlinear + '(loss_item_img_ques[item_ix], dim=1)')

                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat'] and __C.WITH_ANSWER:
                        loss_item_ans[item_ix] = loss_item_ans[item_ix].view(-1)
                    elif loss_nonlinear and __C.WITH_ANSWER:
                        loss_item_ans[item_ix] = eval('F.' + loss_nonlinear + '(loss_item_ans[item_ix], dim=1)')

                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat'] and __C.WITH_ANSWER:
                        loss_item_interp[item_ix] = loss_item_interp[item_ix].view(-1)
                    elif loss_nonlinear and __C.WITH_ANSWER:
                        loss_item_interp[item_ix] = eval('F.' + loss_nonlinear + '(loss_item_interp[item_ix], dim=1)')


                # Now we create all the four losses and then add them
                #print("shape of loss_item_img_ques[0] is {} and of loss_item_img_ques[1] is {}".format(loss_item_img_ques[0],loss_item_img_ques[1]))
                loss_img_ques = loss_fn(loss_item_img_ques[0], loss_item_img_ques[1])

                loss = loss_img_ques
                
                if (__C.WITH_ANSWER):

                    # loss for the prediction from the answer
                    #print("shape of loss_item_ans[0] is {} and of loss_item_ans[1] is {}".format(loss_item_ans[0],loss_item_ans[1]))
                    loss_ans = loss_fn(loss_item_ans[0], loss_item_ans[1])
                
                    # Loss for the prediction from the fused vector
                    # I am keeping the loss same as bce but we can change it later for more predictions
                    # loss_fused = interpolation loss
                    #print("shape of loss_item_interp[0] is {} and of loss_item_interp[1] is {}".format(loss_item_interp[0],loss_item_interp[1]))
                    loss_interp = loss_fn(loss_item_interp[0], loss_item_interp[1])
                    
                    # we also need to multiply this fused loss by a hyperparameter alpha
                    # put the alpha in the config and uncomment the following line
                    loss_interp *= __C.ALPHA
                    loss += loss_ans + loss_interp

                    if (__C.WITH_FUSION_LOSS):

                        # Now calculate the fusion loss
                        #1. Higher loss for higher distance between vectors predicted
                        # by different models for same example

                        dist_calc = (z_img_ques - z_ans).pow(2).sum(1).sqrt()
                        #print("Count of distances being clipped (true is clipped): ", np.unique((dist_calc > __C.CAP_DIST).cpu().numpy(), return_counts=True))

                        '''
                        loss_fusion = torch.min(
                                torch.tensor(__C.CAP_DIST).cuda(),
                                dist_calc
                                ).mean()

                        #2. Lower loss for more distance between two pred vectors of same model
                        loss_fusion -= torch.min(
                                torch.tensor(__C.CAP_DIST).cuda(), 
                                torch.pdist(z_img_ques, 2)
                                ).mean() 

                        loss_fusion -= torch.min(
                                torch.tensor(__C.CAP_DIST).cuda(), 
                                torch.pdist(z_ans, 2)
                                ).mean() 
                        '''

                        loss_fusion = dist_calc.mean()

                        #2. Lower loss for more distance between two pred vectors of same model
                        loss_fusion -= torch.pdist(z_img_ques, 2).mean() 

                        loss_fusion -= torch.pdist(z_ans, 2).mean() 


                        # Multiply the loss fusion with hyperparameter beta
                        loss_fusion *= __C.BETA

                        #print('fusion loss is : {}'.format(loss_fusion))

                        loss +=  loss_fusion

                
                loss /= __C.GRAD_ACCU_STEPS
                loss.backward()

                loss_tmp += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS

            if __C.VERBOSE:
                if dataset_eval is not None:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
                else:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

                print("\r[Version %s][Model %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e" % (
                    __C.VERSION,
                    __C.MODEL_USE,
                    __C.DATASET,
                    epoch + 1,
                    step,
                    int(data_size / __C.BATCH_SIZE),
                    mode_str,
                    loss_tmp / __C.SUB_BATCH_SIZE,
                    optim._rate
                ), end='          ')

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

        wandb.save(
            __C.CKPTS_PATH +
            '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) +
            '.h5'
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

        # if self.__C.VERBOSE:
        #     logfile = open(
        #         self.__C.LOG_PATH +
        #         '/log_run_' + self.__C.VERSION + '.txt',
        #         'a+'
        #     )
        #     for name in range(len(named_params)):
        #         logfile.write(
        #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
        #                 str(name),
        #                 named_params[name][0],
        #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
        #             )
        #         )
        #     logfile.write('\n')
        #     logfile.close()

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
