# WaveSeekerNet: Accurate Prediction of Influenza A Virus Subtypes and Host Source Using Attention-Based Deep Learning


## Introduction

Influenza A virus (IAV) poses a significant threat to animal health globally, with its ability to overcome species barriers and cause pandemics. Rapid and accurate IAV subtype and host source prediction are crucial for effective surveillance and pandemic preparedness. Deep learning has emerged as a powerful tool for analyzing viral genomic sequences, offering new ways to uncover hidden patterns associated with viral characteristics and host adaptation.

We introduce WaveSeekerNet, a novel deep learning model for accurate and rapid prediction of IAV subtypes and host source. The model leverages attention-based mechanisms and efficient token mixing schemes, including the Fourier Transform and the Wavelet Transform, to capture intricate patterns within viral RNA and protein sequences. Extensive experiments on diverse datasets demonstrate WaveSeekerNet’s superior performance to existing models that use the traditional self-attention mechanism. Notably, WaveSeekerNet rivals VADR (Viral Annotation DefineR) in subtype prediction using the high-quality RNA sequences, achieving the maximum score of 1.0 on metrics including the Balanced Accuracy, F1-score (Macro Average), and Matthews Correlation Coefficient (MCC). Our approach to subtype and host source prediction also exceeds the pre-trained ESM-2 (Evolutionary Scale Modeling) models with respect to generalization performance and computational cost. Furthermore, WaveSeekerNet exhibits remarkable accuracy in distinguishing between human, avian, and other mammalian hosts. The ability of WaveSeekerNet to flag potential cross-species transmission events underscores its significant value for real-time surveillance and proactive pandemic preparedness efforts. 

WaveSeekerNet’s superior performance, efficiency, and ability to flag potential cross-species transmission events highlight its potential for real-time surveillance and pandemic preparedness. This model represents a significant advancement in applying deep learning for IAV classification and holds promise for future epidemiological, veterinary studies, and public health interventions. 

## Requirements

1. Pytorch 2.4.1
2. [Pytorch Wavelet package] 1.3.0
3. [Pytorch Optimizer] 3.1.1
4. Other requirements: Python 3.12+, scikit-learn 1.5.1, complexcgr 0.8.0, seaborn 0.13.2, matplotlib 3.9.1, pyfastx 2.1.0, pandas 2.2.2, numpy 1.26.4, biopython 1.84, baycomp 1.0.3.


## Data and Source Code

1. Metadata for the datasets used in the paper can be found in GigaDB (https://doi.org/10.5524/102732).
2. IAV HA and NA RNA/Protein sequences can be downloaded from EpiFLu GISAID database (https://www.gisaid.org/).
3. Source code for model training and evaluation can be found in the `src` directory:
    - `src/WaveSeekerNet`: Contains the WaveSeekerNet code
    - `src/Transformer`: Contains the Transformer-only model and the pre-trained ESM-2 code

## How to Train WaveSeekerNet
To train WaveSeekerNet, follow these steps:

Load dataset
```   
 X_train = np.load(path + 'X_train.npy')
 y_train = np.load(path + 'y_train.npy')
 X_test  = np.load(path + 'X_test.npy')
 y_test  = np.load(path + 'y_test.npy')
```
Parameters for RNA dataset
```  
X_train Shape: (N, 2**D, 2**D) where N is the number of samples, D is the depth of FCGR
D = 6
n_out = len(np.unique(y_train))
n_channels= 1
seq_len = 2**D
res_len = 2**D  
patch_size = (4, 4)
epochs = 35
batch_size = 256
emb_dim = 64
final_hidden_size = 24
 ```  
Parameters for Protein dataset

```  
X_train Shape: (N, 21, seq_len) where N is the number of samples
n_out = len(np.unique(y_train))
n_channels= 1
seq_len = seq_len
res_len = 21
patch_size = (3, res_len)
epochs = 35
batch_size = 256
emb_dim = 64
final_hidden_size = 24
 ```   
WaveSeekerNet Hyperparameters
```     
params_dict = {"use_fft": False,  # default True
               "use_wavelets": False,  # default True
               "use_gmlp": False,  # default True
               "activation_mish": torch.nn.Mish,  # default ErMish
               "activation_gelu": torch.nn.GELU,
               "activation_relu": torch.nn.ReLU,
               "use_kan": False,  # default True
               "use_smoe": False,  # default True
               "use_gc": False,  # default True
               "use_lookahead": False,  # default True
               }
```
Train and predict with WaveSeekerClassifier
```     
clf = WaveSeekerClassifier(
        n_channels=n_channels,
        seq_L=seq_len,
        res_L=res_len,
        patch_size=patch_size,
        n_out=n_out,
        batch_size=batch_size,
        emb_dim=emb_dim,
        final_hidden_size=final_hidden_size,
        epochs=epochs,
        patch_mode="patch",
        wavelet_names=["sym4"],
        n_blocks=1,
        lr=0.0025)
clf.fit(X_train, y_train, X_val, y_val)
clf.predict(X_test)
```



## Contributors and Maintainers

* [Hai Nguyen](https://github.com/nhhaidee) ([CFIA-NCFAD](https://github.com/CFIA-NCFAD), Department of Computer Science, University of Manitoba) - designed the models, wrote the code/manuscript, prepared data, trained models, performed experiments and completed the data analysis.
* [Josip Rudar](https://github.com/jrudar) ([CFIA-NCFAD](https://github.com/CFIA-NCFAD), Department of Integrative Biology & Centre for Biodiversity Genomics, University of Guelph) - designed models, wrote the code, reviewed/edited the manuscript, provided guidance on the project, and provided feedback on the experiments.


[Pytorch Wavelet package]: https://github.com/fbcotter/pytorch_wavelets
[Pytorch Optimizer]:https://github.com/kozistr/pytorch_optimizer