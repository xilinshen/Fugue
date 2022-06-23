import os
import torch
import numpy as np
import math
import glob
import torchvision.transforms as transforms
import tqdm
import natsort

def get_FileSize(file):
    fsize = os.path.getsize(file)
    fsize = fsize/float(1024*1024*1024)
    return round(fsize,2)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        return [q, x]


class RandomSubArrayShuffle(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        
    def __call__(self, x):
        n = len(x)
        idxs = np.array(range(n))
        k = np.int(np.floor(n * self.ratio))
        
        idxs1 = np.random.choice(idxs, k, replace=False)
        idxs2 = np.random.choice(np.setdiff1d(idxs, idxs1), k, replace=False)
        
        a = np.copy(x)
        a[idxs1] = x[idxs2]
        return a

class MoCoDataset_numpy(torch.utils.data.Dataset):
    def __init__(self, expr, batch_label, transform = None):
        #self.expr = torch.tensor(expr)
        self.x = expr
        self.y = torch.tensor(batch_label)
        self.transform = transform
        
    def __getitem__(self,i):
        x = self.x[i]
        
        if self.transform != None:
            x1, x2 = self.transform(x)
            
            return x1, x2, self.y[i]
        else:
            return x, self.y[i]

    def __len__(self):
        return len(self.x)
    
def load_from_numpy(file, shuffle_ratio):
    data = np.load(file)
    expr = data['x']
    batch_label = data['y']
    
    augmentation = [RandomSubArrayShuffle(ratio=shuffle_ratio)]
    
    dataset = MoCoDataset_numpy(expr, batch_label, TwoCropsTransform(transforms.Compose(augmentation))) 
    return dataset

class MoCoDataset_splitfile(torch.utils.data.Dataset):
    def __init__(self, filepath, transform = None):
        self.path = natsort.natsorted(glob.glob(filepath))
        self.transform = transform
        
    def __getitem__(self,i):
        file = self.path[i]
        data = np.load(file)
        x = data['x']
        y = data['y']
        if self.transform != None:
            x1, x2 = self.transform(x)
            return x1, x2, y
        else:
            return x, y

    def __len__(self):
        return len(self.path)

def split_and_load(file, shuffle_ratio, split_now, split_savedir = "./Fugue/Data/example_splitfile/"):
    print("Split data into min-batch and load a few batches of data into memory for each iteration.")
    
    assert split_savedir != None
    
    if not os.path.exists(split_savedir):
        os.mkdir(split_savedir)
        print("Spliting file ...")
    print("split_now", split_now)
    if split_now == True:
        data = np.load(file)
        expr = data['x']
        batch_label = data['y']
    
        # devide each of 32 cells into a file
        np_indexes = []
        np_index = np.arange(len(expr))
        for i in tqdm.tqdm(range(math.floor(len(expr)/32))):
            iter_index = np.random.choice(np_index,32,replace = False)
            np_index = np.setdiff1d(np_index,iter_index)
            iter_expr = expr[iter_index]
            iter_batch = batch_label[iter_index]
            np.savez_compressed("{}/{}.npz".format(split_savedir,i),x = iter_expr, y = iter_batch)
    else:
        print("split_now == False mean that the files has been split into fragments.")

    # dataloader
    augmentation = [RandomSubArrayShuffle(ratio=shuffle_ratio)]
    dataset = MoCoDataset_splitfile("{}/*.npz".format(split_savedir),TwoCropsTransform(transforms.Compose(augmentation)))
    
    return dataset

def load_data(file, shuffle_ratio, load_split_file = False, split_now = False, split_savedir = "./Fugue/Data/example_splitfile/"):
    if load_split_file == True: # if file > 20GB, then split data in 32 cells/a file, and load data.
        dataset = split_and_load(file, shuffle_ratio, split_now, split_savedir)
    else:
        dataset = load_from_numpy(file, shuffle_ratio)
        
    return dataset
