import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score as ba_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, matthews_corrcoef
import random
from model import WaveSeekerClassifier
from sub_modules.fcgr import make_cgr
    
    
def set_seed(random_seed):
    print ("Set Global Seed\n")
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
    
    D=6

    path            = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/Retrain/data/04_HA_Protein_Host/'
    model_save_path = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/Retrain/model_weight/04_HA_Protein_Host/'
    ablation_result = '/fs/vnas_Hcfia/orph/hon000/retrain/results/04_HA_Protein_Host/'

    # train data pre 2020 (good quality)
    X_train = np.load(path + 'X_train_onehot.npy')
    y_train = np.load(path + 'y_train.npy')
    

    # test data post 2020 (good quality)
    X_test_high_quality = np.load(path + 'X_test_high_onehot.npy')
    y_test_high_quality = np.load(path + 'y_test_high.npy')
    
    
    # test data post 2020 (poor quality)
    X_test_low_quality_post2020 = np.load(path + 'X_test_low_onehot.npy')
    y_test_low_quality_post2020 = np.load(path + 'y_test_low.npy')
    
    
    print("Shape:")
    print("Train data shape:", X_train.shape, y_train.shape)
    print("Test High-quality Data Shape:", X_test_high_quality.shape, y_test_high_quality.shape)
    print("Test Low-quality Data Shape (Post 2020):", X_test_low_quality_post2020.shape, y_test_low_quality_post2020.shape)
    
    

    n_out = len(np.unique(y_train))
    seq_len = X_train.shape[2]
    res_len = 21
    patch_size = (3, res_len)
    epochs = 35
    batch_size = 256
    emb_dim = 64
    final_hidden_size = 24
    n_splits = 10

    params_dict = {"use_fft": False,  # default True
                   "use_wavelets": False,  # default True
                   "activation_mish": torch.nn.Mish,  # default ErMish
                   "activation_gelu": torch.nn.GELU,
                   "activation_relu": torch.nn.ReLU,
                   "use_kan": False,  # default True
                   "use_smoe": False,  # default True
                   "use_gc": False,  # default True
                   "use_lookahead": False,  # default True
                   }

    cv_cols = ["Model", "Balanced Accuracy", "F1-Score (Macro)", "Precision (Macro)", "Recall (Macro)", "MCC"]
    param_results_high_quality = []
    param_results_low_quality_post_2020 = []

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=0)
    for kfold_index, (train, test) in enumerate(splitter.split(X_train, y_train)):
        #if kfold_index > 1:
            #break
        print("*************************Fold: ", kfold_index, "**********************************\n")
        clf = WaveSeekerClassifier(
            n_channels=1,
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

        model_name = "Baseline"
        model_weight = model_save_path + "Ablation_weight_" + str(kfold_index) + "_" + model_name + '.pt'

        clf.fit(X_train[train], y_train[train], X_train[test], y_train[test], save_path=model_weight)
        print("%s Result:" % model_name)

        print("High Quality (Post 2020)")
        ctest_high_quality = clf.predict(X_test_high_quality)
        np.save(model_save_path+model_name+"_high_quality_"+str(kfold_index)+".npy",ctest_high_quality)
        
        ba_main, f1_main, p_score_main, r_score_main, mcc_main = get_score(y_test_high_quality, ctest_high_quality)
        param_results_high_quality.append((model_name, ba_main, f1_main, p_score_main, r_score_main, mcc_main))


        print("Low Quality (Post 2020)")
        ctest_test_low_quality_post2020 = clf.predict(X_test_low_quality_post2020)
        np.save(model_save_path+model_name+"_low_quality_"+str(kfold_index)+".npy",ctest_test_low_quality_post2020)
        
        ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2 = get_score(y_test_low_quality_post2020, ctest_test_low_quality_post2020)
        param_results_low_quality_post_2020.append((model_name, ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2))


        del clf
        
        clf = WaveSeekerClassifier(
            n_channels=1,
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
            use_fft=False,
            use_wavelets=False,
            n_blocks=1,
            lr=0.0025)

        model_name = "Baseline_use_wavelets_False_and_use_fft_False"
        model_weight = model_save_path + "Ablation_weight_" + str(kfold_index) + "_" + model_name + '.pt'

        clf.fit(X_train[train], y_train[train], X_train[test], y_train[test], save_path=model_weight)
        print("%s Result:" % model_name)

        print("High Quality (Post 2020)")
        ctest_high_quality = clf.predict(X_test_high_quality)
        np.save(model_save_path+model_name+"_high_quality_"+str(kfold_index)+".npy",ctest_high_quality)
        
        ba_main, f1_main, p_score_main, r_score_main, mcc_main = get_score(y_test_high_quality, ctest_high_quality)
        param_results_high_quality.append((model_name, ba_main, f1_main, p_score_main, r_score_main, mcc_main))


        print("Low Quality (Post 2020)")
        ctest_test_low_quality_post2020 = clf.predict(X_test_low_quality_post2020)
        np.save(model_save_path+model_name+"_low_quality_"+str(kfold_index)+".npy",ctest_test_low_quality_post2020)
        
        ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2 = get_score(y_test_low_quality_post2020, ctest_test_low_quality_post2020)
        param_results_low_quality_post_2020.append((model_name, ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2))


        del clf
        

        for param_name, param_values in params_dict.items():
            clf = WaveSeekerClassifier(
                n_channels=1,
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

            if "activation" in param_name:
                setattr(clf, "activation", param_values)
            else:
                setattr(clf, param_name, param_values)

            model_name = "Baseline_" + param_name + "_" + str(param_values)
            model_weight = model_save_path + "Ablation_weight_" + str(kfold_index) + "_" + model_name + '.pt'

    
            clf.fit(X_train[train], y_train[train], X_train[test], y_train[test], save_path=model_weight)
            print("%s Result:" % model_name)
    
            print("High Quality (Post 2020)")
            ctest_high_quality = clf.predict(X_test_high_quality)
            np.save(model_save_path+model_name+"_high_quality_"+str(kfold_index)+".npy",ctest_high_quality)
            
            ba_main, f1_main, p_score_main, r_score_main, mcc_main = get_score(y_test_high_quality, ctest_high_quality)
            param_results_high_quality.append((model_name, ba_main, f1_main, p_score_main, r_score_main, mcc_main))
    
    
            print("Low Quality (Post 2020)")
            ctest_test_low_quality_post2020 = clf.predict(X_test_low_quality_post2020)
            np.save(model_save_path+model_name+"_low_quality_"+str(kfold_index)+".npy",ctest_test_low_quality_post2020)
            
            ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2 = get_score(y_test_low_quality_post2020, ctest_test_low_quality_post2020)
            param_results_low_quality_post_2020.append((model_name, ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2))

            del clf
            
            if param_name == "use_fft" or param_name == "use_wavelets":
                clf = WaveSeekerClassifier(
                    n_channels=1,
                    seq_L=seq_len,
                    res_L=res_len,
                    patch_size=patch_size,
                    n_out=n_out,
                    batch_size=batch_size,
                    emb_dim=emb_dim,
                    final_hidden_size=final_hidden_size,
                    epochs=epochs,
                    patch_mode="patch",
                    use_gmlp=False,
                    wavelet_names=["sym4"],
                    n_blocks=1,
                    lr=0.0025)
                setattr(clf, param_name, param_values)

                model_name = "Baseline_" + param_name + "_" + str(param_values) + "_and_no_gMLP"
                model_weight = model_save_path + "Ablation_weight_" + str(kfold_index) + "_" + model_name + '.pt'
        
                clf.fit(X_train[train], y_train[train], X_train[test], y_train[test], save_path=model_weight)
                print("%s Result:" % model_name)
        
                print("High Quality (Post 2020)")
                ctest_high_quality = clf.predict(X_test_high_quality)
                np.save(model_save_path+model_name+"_high_quality_"+str(kfold_index)+".npy",ctest_high_quality)
                
                ba_main, f1_main, p_score_main, r_score_main, mcc_main = get_score(y_test_high_quality, ctest_high_quality)
                param_results_high_quality.append((model_name, ba_main, f1_main, p_score_main, r_score_main, mcc_main))
        
        
                print("Low Quality (Post 2020)")
                ctest_test_low_quality_post2020 = clf.predict(X_test_low_quality_post2020)
                np.save(model_save_path+model_name+"_low_quality_"+str(kfold_index)+".npy",ctest_test_low_quality_post2020)
                
                ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2 = get_score(y_test_low_quality_post2020, ctest_test_low_quality_post2020)
                param_results_low_quality_post_2020.append((model_name, ba_test_2, f1_test_2, p_score_test_2, r_score_test_2, mcc_test_2))

                del clf

        df_param_results_high_quality = pd.DataFrame(param_results_high_quality, columns=cv_cols)
        df_param_results_high_quality.to_csv(ablation_result + "Ablation_High_Quality_Fold_" + str(kfold_index) + ".csv")
        print ('\nHigh Quality Post 2020')
        print (df_param_results_high_quality.groupby(['Model'])['F1-Score (Macro)'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print (df_param_results_high_quality.groupby(['Model'])['Balanced Accuracy'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print (df_param_results_high_quality.groupby(['Model'])['MCC'].agg(['mean', 'std', 'sem', 'count']).to_string())
        
        
        df_param_results_low_quality_post_2020 =  pd.DataFrame(param_results_low_quality_post_2020, columns=cv_cols)
        df_param_results_low_quality_post_2020.to_csv(ablation_result + "Ablation_Low_Quality_Fold_" + str(kfold_index) + ".csv")
        print ('\nLow Quality Post 2020')
        print (df_param_results_low_quality_post_2020.groupby(['Model'])['F1-Score (Macro)'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print (df_param_results_low_quality_post_2020.groupby(['Model'])['Balanced Accuracy'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print (df_param_results_low_quality_post_2020.groupby(['Model'])['MCC'].agg(['mean', 'std', 'sem', 'count']).to_string())


if __name__ == '__main__':
    main()
