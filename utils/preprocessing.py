import numpy as np

def logcpm(x):
    x_sum = x.sum(axis = 1) #10000
    x_cpm = (x.transpose()/x_sum)*1e6    
    x_cpm = x_cpm.transpose()
    x_logcpm = np.log2(x_cpm + 1)
    return x_logcpm

def normalize(x_cpm):
    x_mean = x_cpm.mean(axis = 1)
    x_std = x_cpm.std(axis = 1)
    x_cpm_scale = (x_cpm.transpose() - x_mean)/x_std
    return x_cpm_scale.transpose()

def data_preprocessing(count):
    x = logcpm(count)
    x = normalize(x)
    return x