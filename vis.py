import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import argparse
import sys
import copy
from multiprocessing import Pool
from contextlib import contextmanager
import multiprocessing
import os
import time

torch.manual_seed(0)

ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-v", "--version", required=True, type=str, help="folder name in saved/")
ap.add_argument("-e", "--epoch", required=True, type=int, help="epoch number")
ap.add_argument("-n", "--num_samples", required=False, type=int, default=1000, help="number of samples to plot")
ap.add_argument("-t", "--till", required=False, type=bool, default=False, 
    help="Set true to make plots from epoch 0 to --epoch [both included], False to make plots only for epoch number --epoch")

args_parsed = vars(ap.parse_args())

def plotter(X1, X2, args, tsne=False):

    plt.clf()
    filename1 = args['version'] + '/proj_' + str(args['epoch'])
    filename2 = args['version'] + '/ans_' + str(args['epoch'])

    if (tsne):
        filename1 += '_tsne'
        filename2 += '_tsne'
    else:
        filename1 += '_mds'
        filename2 += '_mds'

    red_patch = mpatches.Patch(color='red', label=filename1)
    blue_patch = mpatches.Patch(color='blue', label=filename2)

    x1_min, x1_max = np.min(X1, 0), np.max(X1, 0)
    X1 = (X1 - x1_min) / (x1_max - x1_min)

    x2_min, x2_max = np.min(X2, 0), np.max(X2, 0)
    X2 = (X2 - x2_min) / (x2_max - x2_min)
    
    plt.scatter(X1[:,0], X1[:,1], c='red', s=3, alpha=0.5)
    plt.scatter(X2[:,0], X2[:,1], c='blue', s=3, alpha=0.5)
    plt.legend(handles=[red_patch, blue_patch])

    # plt.show()
    if (tsne):
        plt.savefig('./saved/' + args['version'] + '/vis_' + args['version'] + '_' + str(args['epoch']) + '_tsne' + '.png')
    else:
        plt.savefig('./saved/' + args['version'] + '/vis_' + args['version'] + '_' + str(args['epoch']) + '_mds' + '.png')

def vis_func(args, k=None):

    '''
    args = {
        'version': 'baseline_wa',
        'epoch': 5, #will be replaced by k (if passed)
        'num_samples': 1000,
        'till': True
    }
    '''

    if (k is not None):
        args['epoch'] = k

    filename1 = './saved/' + args['version'] + '/z_proj_' + str(args['epoch']) + '.npy'
    filename2 = './saved/' + args['version'] + '/z_ans_' + str(args['epoch']) + '.npy'

    print("Started for epoch %d with pid %d: " % (args['epoch'], os.getpid()))

    if (args['num_samples'] is not None):
        num_samples = int(args['num_samples'])

    else:
        num_samples = None

    #print("Loading file 1")
    z1_load = np.load(filename1)
    z1_load = torch.from_numpy(z1_load)
    #print("z1 shape: ", z1_load.shape)

    #print("Loading file 2")
    z2_load = np.load(filename2)
    z2_load = torch.from_numpy(z2_load)
    #print("z2 shape: ", z2_load.shape)

    embedding = MDS(n_components=2)
    tsne_embedding = TSNE()

    #print("Transforming z1, z2")
    if (num_samples is None):
        z1_transformed = embedding.fit_transform(z1_load)
        z2_transformed = embedding.fit_transform(z2_load)
        
        z1_transformed_tsne = embedding.fit_transform(z1_load)
        z2_transformed_tsne = embedding.fit_transform(z2_load)

    else:
        z1_transformed = embedding.fit_transform(z1_load[:num_samples])
        z2_transformed = embedding.fit_transform(z2_load[:num_samples])
        
        z1_transformed_tsne = embedding.fit_transform(z1_load[:num_samples])
        z2_transformed_tsne = embedding.fit_transform(z2_load[:num_samples])

    plotter(z1_transformed, z2_transformed, args)
    print("Mds Image save successful for: ", args['epoch'])

    plotter(z1_transformed_tsne, z2_transformed_tsne, args, tsne=True)
    print("Tsne Image save successful for: ", args['epoch'])

def vis_func_unpacker(args):
    vis_func(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


if __name__ == '__main__':
    start = time.time()
    if (args_parsed['till'] is True):

        dic = copy.deepcopy(args_parsed)
        n = int(dic['epoch'])

        with poolcontext(processes= n+1 ) as p:
            p.map(vis_func_unpacker, [(dic, i) for i in range(n+1)])

    else:
        vis_func(args_parsed)
    end = time.time()
    print("Time taken (in s): ", end-start)
