import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score as ba_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, matthews_corrcoef
from transformer_based import TransfomerClassifier
from sampling import get_rare_sequence, resampling

import random


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_score(y_true, y_pred):
    ba = ba_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    p_score = precision_score(y_true, y_pred, average="macro")
    r_score = recall_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    print(ba, f1, p_score, r_score, mcc)
    print(classification_report(y_true, y_pred, zero_division=np.nan))
    return ba, f1, p_score, r_score, mcc


def main():
    set_seed(0)
    
    train_path      = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/Retrain/data/Full_Train_Data/04_HA_RNA/'
    path            = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/Retrain/data/04_HA_RNA_Subtype/' # no change
    ablation_result = '/fs/vnas_Hcfia/orph/hon000/revision_1/transformer/result/04_HA_RNA_Subtype/'
    model_weights   = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/Revision_1/model_weight/transformer/04_HA_RNA_Subtype/'

    # train data pre 2020 (good quality)
    X_train = np.load(train_path + 'X_train_fcgr.npy')
    y_train = np.load(train_path + 'y_train_subtype.npy')

    # test data post 2020 (good quality)
    X_test_high_quality = np.load(path + 'X_test_high_fcgr.npy')
    y_test_high_quality = np.load(path + 'y_test_high.npy')

    # test data post 2020 (poor quality)
    X_test_low_quality = np.load(path + 'X_test_low_fcgr.npy')
    y_test_low_quality = np.load(path + 'y_test_low.npy')

    print("OneHot training data shape:", X_train.shape, y_train.shape)
    print("Test High-quality Data Shape:", X_test_high_quality.shape, y_test_high_quality.shape)
    print("Test Low-quality Data Shape (Post 2020):", X_test_low_quality.shape, y_test_low_quality.shape)

    n_out = len(np.unique(y_train))
    print("Number of classes:", n_out)
    patch_size= (4, 4)
    epochs = 150
    batch_size = 256
    n_splits = 10
    n_channels= 1


    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=0)
    cv_cols = ["Model", "Balanced Accuracy", "F1-Score (Macro)", "Precision (Macro)", "Recall (Macro)", "MCC"]
    param_results_high_quality = []
    param_results_low_quality = []
    
    X_rare, y_rare, X_CV, y_CV = get_rare_sequence(X_train, y_train, s_splits=n_splits, n_samples=600) # For subtype only
    
    for kfold_index, (train, test) in enumerate(splitter.split(X_CV, y_CV)):
        print("************Fold:", kfold_index)
        
        #Sampling in training loop
        X_CV_train, y_CV_train = resampling(X_CV[train], y_CV[train], n_downsamples=6000, n_upsamples=600)
        
        # Concatenate with rare subtypes (rare for training only which was only sample before CV)
        X_CV_train = np.concatenate((X_CV_train, X_rare), axis =0)
        y_CV_train = np.concatenate((y_CV_train, y_rare), axis =0)
        
        X_CV_test  = X_CV[test]
        y_CV_test  = y_CV[test]
        
        for use_fnet in [True, False]:
            for emb_dim in [64, 128]:

                print('Emb Dim:', emb_dim)

                if use_fnet == True:
                    model_name = "Transformer-FNET-" + str(emb_dim)
                else:
                    model_name = "Transformer-MHA-" + str(emb_dim)
                print(model_name)

                clf = TransfomerClassifier(emb_dim, n_out, "patch", patch_size, batch_size, epochs, n_channels=n_channels,
                                           use_fnet=use_fnet, save_path=model_weights + model_name + '_' + str(kfold_index) + ".pt")

                clf.fit(X_CV_train, y_CV_train, X_CV_test, y_CV_test)

                y_pred_high_quality = clf.predict(X_test_high_quality, y_test_high_quality)
                np.save(model_weights + model_name + '_y_pred_high_quality_' + str(kfold_index) + '.npy', y_pred_high_quality)
                y_pred_low_quality = clf.predict(X_test_low_quality, y_test_low_quality)
                np.save(model_weights + model_name + '_y_pred_low_quality_' + str(kfold_index) + '.npy', y_pred_low_quality)

                print('High Quality')
                ba_main, f1_main, p_score_main, r_score_main, mcc = get_score(y_test_high_quality, y_pred_high_quality)
                param_results_high_quality.append((model_name, ba_main, f1_main, p_score_main, r_score_main, mcc))

                print("Low Quality")
                ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_2 = get_score(y_test_low_quality, y_pred_low_quality)
                param_results_low_quality.append((model_name, ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_2))

                del clf
                
        df_high_quality = pd.DataFrame(param_results_high_quality, columns=cv_cols)
        print('\nHigh Quality Post 2020')
        print(df_high_quality.groupby(['Model'])['F1-Score (Macro)'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print(df_high_quality.groupby(['Model'])['Balanced Accuracy'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print(df_high_quality.groupby(['Model'])['MCC'].agg(['mean', 'std', 'sem', 'count']).to_string())

    
        df_low_quality= pd.DataFrame(param_results_low_quality, columns=cv_cols)
        print('\nLow Quality Post 2020')
        print(df_low_quality.groupby(['Model'])['F1-Score (Macro)'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print(df_low_quality.groupby(['Model'])['Balanced Accuracy'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print(df_low_quality.groupby(['Model'])['MCC'].agg(['mean', 'std', 'sem', 'count']).to_string())
    
        print("Saving Results")
        df_high_quality.to_csv(ablation_result+"Transformer_High_Quality_"+ str(kfold_index)+".csv")
        df_low_quality.to_csv(ablation_result+"Transformer_Low_Quality_"+ str(kfold_index)+".csv")

if __name__ == '__main__':
    main()
