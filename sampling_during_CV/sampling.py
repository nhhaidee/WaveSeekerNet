import numpy as np
from sklearn.utils import resample
import random

def get_rare_sequence(X, y, s_splits = 10, n_samples=600):
    
    unique, counts = np.unique(y, return_counts=True)
    rare_subtype_counts_index      = np.where(counts < s_splits, True, False) # get index where num_sybtype < 10
    rare_subtypes                  = unique[rare_subtype_counts_index] # get subtypes name that has num_sybtype < 10
    rare_subtypes_counts           = counts[rare_subtype_counts_index]

    for i, s_type in enumerate(rare_subtypes):
        s_type_index    = np.where(y == s_type, True, False)
        X_rare_subtype  = X[s_type_index]
        y_rare_subtype  = y[s_type_index]

        X_rare_subtype, y_rare_subtype = resample(X_rare_subtype, y_rare_subtype, replace=True, n_samples=n_samples, random_state=0)

        if i==0:
            X_temp = X_rare_subtype
            y_temp = y_rare_subtype
        else:
            X_temp = np.concatenate((X_temp, X_rare_subtype), axis=0)
            y_temp = np.concatenate((y_temp, y_rare_subtype), axis=0)

    s_type_other_index = np.where(np.isin(y, rare_subtypes), True, False)
    X_other_subtype    = X[~s_type_other_index]
    y_other_subtype    = y[~s_type_other_index]

    print ("Rare subtype information after sampling")
    print (X_temp.shape, y_temp.shape)
    print (np.transpose(np.unique(y_temp, return_counts=True)))

    print ("Other subtype information without sampling")
    print (X_other_subtype.shape, y_other_subtype.shape)
    print (np.transpose(np.unique(y_other_subtype, return_counts=True)))

    return X_temp, y_temp, X_other_subtype, y_other_subtype


def resampling(X, y, n_downsamples=6000, n_upsamples=600):
    unique, counts = np.unique(y, return_counts=True)
    print ("Data Shape Before Sampling: ", X. shape, y.shape)
    for i, ele in enumerate(unique):
        n_counts = counts[i]
        if n_counts > n_downsamples:
            ele_index = np.where(y == ele, True, False)
            X_ele = X[ele_index]
            y_ele = y[ele_index]
            X_ele, y_ele = resample(X_ele, y_ele, replace=False, n_samples=n_downsamples, random_state=0)
            print ("Subtype/Host:", ele, ",count: ", n_counts, ",index count:", np.unique(ele_index, return_counts=True), ",downsampling: ", X_ele.shape, y_ele.shape)
        elif n_counts < n_upsamples:
            ele_index = np.where(y == ele, True, False)
            X_ele = X[ele_index]
            y_ele = y[ele_index]
            X_ele, y_ele = resample(X_ele, y_ele, replace=True, n_samples=n_upsamples, random_state=0)
            print ("Subtype/Host:", ele, ",count: ", n_counts, ",index count:", np.unique(ele_index, return_counts=True), ",upsampling: ", X_ele.shape, y_ele.shape)
        else:
            ele_index = np.where(y == ele, True, False)
            X_ele = X[ele_index]
            y_ele = y[ele_index]
            print ("Subtype/Host:", ele, ",count: ", n_counts, ",index count:", np.unique(ele_index, return_counts=True), ",keep: ", X_ele.shape, y_ele.shape)
        
        if i == 0:
            X_temp = X_ele
            y_temp = y_ele
        else:
            X_temp = np.concatenate((X_temp, X_ele), axis=0)
            y_temp = np.concatenate((y_temp, y_ele), axis=0)

    return X_temp, y_temp
        