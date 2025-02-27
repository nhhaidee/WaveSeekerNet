# WaveSeekerNet: Accurate Prediction of Influenza A Virus Subtypes and Host Source Using Attention-Based Deep Learning


## Introduction

Influenza A virus (IAV) poses a significant threat to animal health globally, with its ability to overcome species barriers and cause pandemics. Rapid and accurate prediction of IAV subtypes and host source is crucial for effective surveillance and pandemic preparedness. Deep learning has emerged as a powerful tool for analyzing viral genomic sequences, offering new ways to uncover hidden patterns associated with viral characteristics and host adaptation.

We introduce WaveSeekerNet, a novel deep learning model for accurate and rapid prediction of IAV subtypes and host source. The model leverages attention-based mechanisms and efficient token mixing schemes, including the Fast Fourier Transform and the Wavelet Transform, to capture intricate patterns within viral RNA and protein sequences. Extensive experiments on diverse datasets demonstrate WaveSeekerNet's superior performance compared to existing Transformer-only models. Notably, WaveSeekerNet achieves scores of up to the maximum 1.0 across all evaluation metrics, including F1-score (Macro Average), Balanced Accuracy and Matthews Correlation Coefficient (MCC), in subtype prediction, even for rare subtypes. Furthermore, WaveSeekerNet exhibits remarkable accuracy in distinguishing between human, avian, and other mammalian hosts. The ability of WaveSeekerNet to flag potential cross-species transmission events underscores its significant value for real-time surveillance and proactive pandemic preparedness efforts.

This repository contains the source code and data used to train WaveSeekerNet. The paper is under review and will be made available upon acceptance.
## Requirements

1. Pytorch 2.4.1
2. [Pytorch Wavelet package] 1.3.0
3. [Pytorch Optimizer] 3.1.1
4. Other requirements: Python 3.12+, scikit-learn 1.5.1, complexcgr 0.8.0, seaborn 0.13.2, matplotlib 3.9.1, pyfastx 2.1.0, pandas 2.2.2, numpy 1.26.4, biopython 1.84, baycomp 1.0.3.


## Data and Source Code

1. Metadata for the datasets used in the paper can be found in the `data` directory.
2. IAV HA and NA RNA/Protein sequences can be downloaded from EpiFLu GISAID database (https://www.gisaid.org/).
3. Source code for model training and evaluation can be found in the `src` directory. The experimental results and training logs are alo available in the `src` directory with extension `.csv` and `.out` respectively.


[Pytorch Wavelet package]: https://github.com/fbcotter/pytorch_wavelets
[Pytorch Optimizer]:https://github.com/kozistr/pytorch_optimizer