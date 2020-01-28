
# coding: utf-8

# In[85]:


import numpy as np
import torch
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("macOSX")
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
ap.add_argument("-f1", "--filename1", required=True,
   help="first npy filename")
ap.add_argument("-f2", "--filename2", required=True,
   help="second npy filename")
ap.add_argument("-n", "--num_samples", required=False,
   help="number of samples to plot")

args = vars(ap.parse_args())

filename1 = args['filename1']
filename2 = args['filename2']
if (args['num_samples'] is not None):
    num_samples = int(args['num_samples'])
else:
    num_samples = None

'''
print(filename1, type(filename1))
print(filename2, type(filename2))
print(num_samples, type(num_samples))
'''

# # Load the NP array from disk, convert to torch tensor

# In[25]:


print("Loading file 1")
z1_load = np.load(filename1)
z1_load = torch.from_numpy(z1_load)


# In[26]:


print("Loading file 2")
z2_load = np.load(filename2)
z2_load = torch.from_numpy(z2_load)


# # MDS Visualisation

# In[31]:


embedding = MDS(n_components=2)


# In[32]:


print("Transforming z1 and z2")
if (num_samples is None):
    z1_transformed = embedding.fit_transform(z1_load)
    z2_transformed = embedding.fit_transform(z2_load)
else:
    z1_transformed = embedding.fit_transform(z1_load[:num_samples])
    z2_transformed = embedding.fit_transform(z2_load[:num_samples])


def plotter(X1, X2):
    x1_min, x1_max = np.min(X1, 0), np.max(X1, 0)
    X1 = (X1 - x1_min) / (x1_max - x1_min)

    x2_min, x2_max = np.min(X2, 0), np.max(X2, 0)
    X2 = (X2 - x2_min) / (x2_max - x2_min)
    
    plt.scatter(X1[:,0], X1[:,1], c='r')
    plt.scatter(X2[:,0], X2[:,1], c='b')

    # plt.show()
    plt.savefig('vis.png')


# In[83]:

print("Plotting and Saving Points")
plotter(z1_transformed, z2_transformed)

# In[74]:

