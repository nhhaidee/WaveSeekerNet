from sklearn.metrics import balanced_accuracy_score as ba_score
import torch 
from torch import nn
import math
from torch.optim import AdamW
import numpy as np
from time import time
from torchinfo import summary
import esm
from early_stopping_pytorch import EarlyStopping
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TransformerBlock(nn.Module):
    def __init__(self, num_classes, embedding_dim=480):
        super(TransformerBlock, self).__init__()
        self.num_classes = num_classes

        self.embedding_dim = embedding_dim


        self.transformer, _ = esm.pretrained.load_model_and_alphabet("/fs/vnas_Hcfia/orph/hon000/esm_model/esm2_t12_35M_UR50D.pt")

        self.cls_head = nn.Sequential(
            nn.Linear(embedding_dim, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 30),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(30, self.num_classes)
        )

    def pooler_fn(
        self, 
        token_embeddings, 
    ): 

        sum_embeddings = torch.sum(token_embeddings, 1)
        sum_mask = token_embeddings.size(1)
        output_vector = sum_embeddings / sum_mask

        return output_vector      

    def forward(self,inputs):
        x = self.transformer(inputs, repr_layers=[12])#, return_contacts=True)
        x = x["representations"][12]
        x = self.pooler_fn(token_embeddings = x)
        x = self.cls_head(x)

        return x 

class TransformerClassifier():
    def __init__(self, emb_dim, n_out, batch_size, epochs, save_path='checkpoint.pt'):
        self.emb_dim = emb_dim
        self.n_out = n_out
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_path = save_path

    def fit(self, X, y, X_va, y_va):

        self.clf = TransformerBlock(self.n_out, self.emb_dim).to("cuda")
        
        logging.info("Start Training Model")

#        summary(self.clf, input_size=(2, X.shape[1]),
#        col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"], depth=5)

        X_tr = torch.tensor(X).to("cuda")
        y_tr = torch.tensor(y).to("cuda")
    
        X_valid = torch.tensor(X_va).to("cuda")
        y_valid = torch.tensor(y_va).to("cuda")
        
        # Create a dataset loader
        dataset_train = torch.utils.data.DataLoader(
                list(zip(X_tr, y_tr)),
                shuffle=True,
                batch_size=self.batch_size,
            )
    
        dataset_valid = torch.utils.data.DataLoader(
                list(zip(X_valid, y_valid)), shuffle=False, batch_size=self.batch_size
            )
                
        model_parameters = filter(lambda p: p.requires_grad, self.clf.parameters())
        params_train = sum([np.prod(p.size()) for p in model_parameters])
        print("Total Trainable Parameters: ", params_train)
    
        model_parameters = self.clf.parameters()
        params_total = sum([np.prod(p.size()) for p in model_parameters])
        print("Total Parameters: ", params_total)
    
        # Prepare optimizer and loss function
        param_optimizer = list(self.clf.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_parameters, lr=1e-4, eps=1e-8)
    
        loss_fn = torch.nn.CrossEntropyLoss().to("cuda")
    
        scaler = torch.amp.GradScaler("cuda")
        
        early_stopping = EarlyStopping(patience=7, delta=0.001, path=self.save_path, verbose=True)
        
        for epoch in range(self.epochs):
            self.clf.train()
            start = time()
            for batch_num, batch in enumerate(dataset_train):
                x_in, y_in = batch
                with torch.amp.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
    
                    outputs = self.clf(x_in)
    
                    loss_v = loss_fn(outputs, y_in)
    
                optimizer.zero_grad()
                scaler.scale(loss_v).backward()
                scaler.step(optimizer)
                scaler.update()
            
            end = time()
            total_training_time = end - start
            
            self.clf.eval()
            val_pred = []
            val_labels = []
            start = time()
            
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in dataset_valid:
                    x_v, y_v = batch
        
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        outputs = self.clf(x_v)
        
                        pred = torch.max(outputs, 1)[1]
                    
                        val_pred.extend(pred.cpu().numpy())
                        val_labels.extend(y_v.cpu().numpy())
                        
                        loss_validation = loss_fn(outputs, y_v)
                        val_loss += loss_validation.item()
                        
            end = time()
            total_val_infer_time = end - start
            val_loss /= len(dataset_valid)
            logging.info("Train epoch %d in %f seconds", epoch+1, total_training_time)
            print(f"----Epoch:{epoch+1}----")
            print("Training Time :", total_training_time)
            print("Val Infer Time:", total_val_infer_time)
            print("BAS Score on Validation Data:", ba_score(val_labels, val_pred))
            print("Loss on Validation Data:", val_loss)
            
            early_stopping(val_loss, self.clf)
            if early_stopping.early_stop:
                print("Early Stop")
                break
                
        #if self.save_path is not None:
        #    torch.save(self.clf.state_dict(), self.save_path)
        print ("Load the best weight after training:", self.save_path)
        logging.info("Finish training and load weight for prediction:%s", self.save_path)
        self.clf.load_state_dict(torch.load(self.save_path, weights_only=True))
        return self

    def predict(self, X, y):
        
        test_pred = []

        X_te = torch.tensor(X).to("cuda")
        y_te = torch.tensor(y).to("cuda")
        
        dataset_test = torch.utils.data.DataLoader(
            list(zip(X_te, y_te)), shuffle=False, batch_size=self.batch_size
        )

        self.clf.eval()
        start = time()
        with torch.no_grad():
            for batch in dataset_test:
                x_v, _ = batch
    
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    outputs = self.clf(x_v)
    
                    pred = torch.max(outputs, 1)[1]
    
                    test_pred.extend(pred.cpu().numpy())
        end = time()
        total_time = end - start
        print("Inference Test Time:", total_time)
        test_pred = np.asarray(test_pred)

        
        return test_pred


