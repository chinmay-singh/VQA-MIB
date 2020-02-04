
# coding: utf-8

# In[85]:


import numpy as np
import torch
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import argparse
import sys

# Set random seed for Pytorch
torch.manual_seed(0)


# # Create a random torch tensor, convert to NP array, save to disk

# In[21]:


'''
z1 = torch.randn((1000, 1024))
z1 = z1.numpy()
print(z1)
filename = 'z1.npy'
np.save(filename, z1)
'''


# In[22]:


'''
z2 = torch.randn((1000, 1024))
z2 = z2.numpy()
print(z2)
filename = 'z2.npy'
np.save(filename, z2)
'''


# # Parse arguments from command line

# In[86]:


ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-v", "--version", required=True,
        help="folder name in saved/")
ap.add_argument("-e", "--epoch", required=True,
        help="epoch number")
ap.add_argument("-n", "--num_samples", required=False,
   help="number of samples to plot")

args = vars(ap.parse_args())

filename1 = './saved/' + args['version'] + '/z_proj_' + args['epoch'] + '.npy'
filename2 = './saved/' + args['version'] + '/z_ans_' + args['epoch'] + '.npy'
filename3 = './saved/' + args['version'] + '/z_fused_' + args['epoch'] + '.npy'

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

#print("Loading file 3")
z3_load = np.load(filename3)
z3_load = torch.from_numpy(z3_load)
#print("z3 shape: ", z3_load.shape)

embedding = MDS(n_components=2)

#print("Transforming z1, z2 and z3")
if (num_samples is None):
    z1_transformed = embedding.fit_transform(z1_load)
    z2_transformed = embedding.fit_transform(z2_load)
    z3_transformed = embedding.fit_transform(z3_load)
else:
    z1_transformed = embedding.fit_transform(z1_load[:num_samples])
    z2_transformed = embedding.fit_transform(z2_load[:num_samples])
    z3_transformed = embedding.fit_transform(z3_load[:num_samples])


def plotter(X1, X2, X3):

    red_patch = mpatches.Patch(color='red', label=filename1)
    blue_patch = mpatches.Patch(color='blue', label=filename2)
    green_patch = mpatches.Patch(color='green', label=filename3)

    x1_min, x1_max = np.min(X1, 0), np.max(X1, 0)
    X1 = (X1 - x1_min) / (x1_max - x1_min)

    x2_min, x2_max = np.min(X2, 0), np.max(X2, 0)
    X2 = (X2 - x2_min) / (x2_max - x2_min)
    
    x3_min, x3_max = np.min(X3, 0), np.max(X3, 0)
    X3 = (X3 - x3_min) / (x3_max - x3_min)

    plt.scatter(X1[:,0], X1[:,1], c='red', s=3, alpha=0.5)
    plt.scatter(X2[:,0], X2[:,1], c='blue', s=3, alpha=0.5)
    plt.scatter(X3[:,0], X3[:,1], c='green', s=3, alpha=0.5)
    plt.legend(handles=[red_patch, blue_patch, green_patch])

    # plt.show()
    plt.savefig('./saved/' + args['version'] + '/vis_' + args['version'] + '_' + args['epoch'] + '.png')


##print("Plotting and Saving Points")
plotter(z1_transformed, z2_transformed, z3_transformed)
print("Image save successful for: ", args['epoch'])

# In[74]:

