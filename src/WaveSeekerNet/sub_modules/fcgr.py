from complexcgr import FCGR
import numpy as np

def make_cgr(x, D=6):

   
    fcgr = FCGR(k=D)
   
    seqs = []
    for seq in x:
   
        tmp = seq.transpose()
        padding = np.sum(tmp, axis=1)
        padding = padding > 0
        
        tmp = tmp[padding, :]
   
        tmp = np.argmax(tmp, axis = 1)
   
        tmp = np.where(tmp == 0, "A", tmp)
        tmp = np.where(tmp == "1", "C", tmp)
        tmp = np.where(tmp == "2", "G", tmp)
        tmp = np.where(tmp == "3", "T", tmp)
        tmp = np.where(tmp == "4", "N", tmp)
   
        tmp = "".join(tmp)
       
        tmp = fcgr(tmp)

        max_sz = 4**D
        fcgr_sum = np.sum(tmp)
       
        tmp = tmp / fcgr_sum
        tmp = tmp * max_sz
   
        seqs.append(tmp)

    seqs = np.asarray(seqs)

    return seqs
    
def make_cgr_multi_channels(y, D =6):

    for i, seqs in enumerate(y):
        cgr_sample = make_cgr(seqs, D)
        cgr_sample = np.expand_dims(cgr_sample, axis=0)
        if i%5000==0:
            print("Done:", i)
        if i == 0:
            tmp = cgr_sample
        else:
            tmp = np.vstack((tmp, cgr_sample))
    
    return tmp