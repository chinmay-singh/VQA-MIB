# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.models.model_loader import CfgLoader
from utils1.exec import Execution 
import argparse, yaml


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='OpenVQA Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      default='train',
                      type=str, required=False)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=[
                           'mcan_small',
                           'mcan_large',
                           'ban_4',
                           #Edits
                           'ban_wa',
                           'baseline_wa',
                           #End of Edits
                           'ban_8',
                           'mfb',
                           'mfh',
                           'mem',
                           'butd',
                           'baseline',
                           'baseline_wa_no_fusion',
                           'positional',
                           'mcan_large_wa'
                           ]
                        ,
                      help='{'
                           'mcan_small,'
                           'mcan_large,'
                            #Edits
                           'ban_wa,'
                           'baseline_wa,'
                           #End of Edits
                           'ban_4,'
                           'ban_8,'
                           'mfb,'
                           'mfh,'
                           'butd,'
                           'baseline,'
                           'baseline_wa_no_fusion,'
                           'positional,'
                           '}'
                        ,
                      default='baseline_wa',
                      type=str, required=False)

    parser.add_argument('--DATASET', dest='DATASET',
                      choices=['vqa', 'gqa', 'clevr'],
                      help='{'
                           'vqa,'
                           'gqa,'
                           'clevr,'
                           '}'
                        ,
                      default='vqa',  
                      type=str, required=False)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "vqa: {'train', 'train+val', 'train+val+vg'}"
                           "gqa: {'train', 'train+val'}"
                           "clevr: {'train', 'train+val'}"
                        ,
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      choices=['True', 'False'],
                      help='True: evaluate the val split when an epoch finished,'
                           'False: do not evaluate on local',
                      type=str)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      choices=['True', 'False'],
                      help='True: save the prediction vectors,'
                           'False: do not save the prediction vectors',
                      type=str)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size in training',
                      type=int)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu choose, eg.'0, 1, 2, ...'",
                      default='0, 1',
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      default='baseline_wa_sweep',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      choices=['True', 'False'],
                      help='True: use checkpoint to resume training,'
                           'False: start training with random init',
                      type=str)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead, it will override'
                           'CKPT_VERSION and CKPT_EPOCH',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='split batch to reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading to accelerate IO',
                      type=int)

    parser.add_argument('--CAP_DIST', dest='CAP_DIST',
                      help='Capping the Euclidean distance for the Fusion',
                      default=0.3,
                      type=float)

    parser.add_argument('--ALPHA', dest='ALPHA',
                      help='ALPHA: Combining parameter for interpolation Loss',
                      default=1.0,
                      type=float)

    parser.add_argument('--BETA', dest='BETA',
                      help='BETA: Combining parameter for interpolation Loss',
                      default=30.0,
                      type=float)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      choices=['True', 'False'],
                      help='True: use pin memory, False: not use pin memory',
                      type=str)

    parser.add_argument('--VERB', dest='VERBOSE',
                      choices=['True', 'False'],
                      help='True: verbose print, False: simple print',
                      type=str)

    parser.add_argument('--PROJ_STDDEV', dest='PROJ_STDDEV',
                      help='ALPHA: Combining parameter for interpolation Loss',
                      default=1.0,
                      type=float)

    parser.add_argument('--ANS_STDDEV', dest='ANS_STDDEV',
                      help='ALPHA: Combining parameter for interpolation Loss',
                      default=1.0,
                      type=float)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                          help='ALPHA: Combining parameter for interpolation Loss',
                          default=13,
                          type=int)


    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    cfg_file = "configs/{}/{}.yml".format(args.DATASET, args.MODEL)
    with open(cfg_file, 'r') as f:

        # Loads the yaml file
        yaml_dict = yaml.load(f)

    # Loads the model_cfgs + base_cfgs
    __C = CfgLoader(yaml_dict['MODEL_USE']).load()

    # Loads the command line cfgs
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)

    # {**dict1, **dict2} creates a new dictionary by merging dict1 and dict2, using dict2 for key clashes
    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    # FINAL PREFERENCE OF CFGS:
    # COMMAND LINE > YAML FILE > MODEL CFGS > BASE CFGS

    print('Hyper Parameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)




