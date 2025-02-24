# WaveSeekerNet: Accurate Prediction of Influenza A Virus Subtypes and Host Source Using Attention-Based Deep Learning


## Introduction

Influenza A virus (IAV) poses a significant threat to animal health globally, with its ability to overcome species barriers and cause pandemics. Rapid and accurate prediction of IAV subtypes and host source is crucial for effective surveillance and pandemic preparedness. Deep learning has emerged as a powerful tool for analyzing viral genomic sequences, offering new ways to uncover hidden patterns associated with viral characteristics and host adaptation.

We introduce WaveSeekerNet, a novel deep learning model for accurate and rapid prediction of IAV subtypes and host source. The model leverages attention-based mechanisms and efficient token mixing schemes, including the Fast Fourier Transform and the Wavelet Transform, to capture intricate patterns within viral RNA and protein sequences. Extensive experiments on diverse datasets demonstrate WaveSeekerNet's superior performance compared to existing Transformer-only models. Notably, WaveSeekerNet achieves scores of up to the maximum 1.0 across all evaluation metrics, including F1-score (Macro Average), Balanced Accuracy and Matthews Correlation Coefficient (MCC), in subtype prediction, even for rare subtypes. Furthermore, WaveSeekerNet exhibits remarkable accuracy in distinguishing between human, avian, and other mammalian hosts. The ability of WaveSeekerNet to flag potential cross-species transmission events underscores its significant value for real-time surveillance and proactive pandemic preparedness efforts.

This repository contains the source code and data used to train WaveSeekerNet. The code is written in Python and PyTorch, and is available for research purposes. For more details, please refer to our paper:

```Rudar J, Kruczkiewicz P, Vernygora O, Golding GB, Hajibabaei M, Lung O. Sequence signatures within the genome of SARS-CoV-2 can be used to predict host source. Microbiol Spectr 2024;12:e03584-23. https://doi.org/10.1128/spectrum.03584-23.```

## Requirements

1. Pytorch 2.4.0 or higher
2. [Pytorch Wavelet package]
3. [Pytorch Optimizer]

## Data

1. Metadata for the datasets used in the paper can be found in the `data` directory.
2. IAV HA and NA RNA/Protein sequences can be downloaded from EpiFLu GISAID database (https://www.gisaid.org/).



[Pytorch Wavelet package]: https://github.com/fbcotter/pytorch_wavelets
[Pytorch Optimizer]:pytorch_optimizer