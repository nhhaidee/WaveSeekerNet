from sklearn.metrics import balanced_accuracy_score as ba_score
import torch 
from torch import nn
import math
from torch.optim import AdamW
import numpy as np
from time import time
from torchinfo import summary

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        """
        From: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        """
        # inherit from Module
        super().__init__()

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        return x

class MakePatches(nn.Module):
    def __init__(self, patch_width, patch_height, stride):
        super().__init__()

        self.unfold = nn.Unfold((patch_width, patch_height), stride=stride)

    def forward(self, inputs):
        if len(inputs.shape)==3:
            x_in = inputs.unsqueeze(1)

            x = self.unfold(x_in)
            x = torch.transpose(x, -1, 1)
        else:
            x = self.unfold(inputs)
            x = torch.transpose(x, -1, 1)

        return x

class FNET(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.norm_skip = nn.LayerNorm(emb_dim)
     
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.125),
            nn.Linear(emb_dim, emb_dim)
        )
     
        self.norm_out = nn.LayerNorm(emb_dim)

    def forward(self, x_p):
        
        x_fft = torch.real(torch.fft.fft2(x_p.float(), norm="forward"))
         
        x_fft = x_fft + x_p
         
        x_fft = self.norm_skip(x_fft)
         
        x_fft = self.ffn(x_fft) + x_fft
         
        x_fft = self.norm_out(x_fft)
         
        return x_fft

class TransfomerBlock(nn.Module):
    def __init__(self, patch_size, num_classes, embedding_dim=128, patch_mode="patch", n_channels=1, use_fnet=False):
        super(TransfomerBlock, self).__init__()
        self.num_classes = num_classes
        self.patch_width = patch_size[1]
        self.patch_height = patch_size[0]
        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        self.use_fnet=use_fnet

        if patch_mode == "patch" or patch_mode == "compress":
            self.stride = self.patch_height
        else:
            self.stride = 1

        self.embedding = nn.Linear(self.n_channels*self.patch_width*self.patch_height, self.embedding_dim )
        if use_fnet == False:
            self.transformer = nn.TransformerEncoderLayer(d_model=self.embedding_dim , nhead=4)
        else:
            self.fnet= FNET(self.embedding_dim)
        
        #self.transformer = nn.TransformerEncoder(transformer, num_layers = 3)
        self.cls_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        )

        self.make_patches = MakePatches(self.patch_width, self.patch_height, self.stride)
        self.positional_encoding = PositionalEncoding(self.embedding_dim)


    def pooler_fn(
        self, 
        token_embeddings, 
    ): 

        sum_embeddings = torch.sum(token_embeddings, 1)
        sum_mask = token_embeddings.size(1)
        output_vector = sum_embeddings / sum_mask

        return output_vector      

    def forward(
        self,
        inputs,
    ):
        x = self.make_patches(inputs)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        if self.use_fnet == False:
            x = self.transformer(x)
        else:
            x = self.fnet(x)
        x = self.pooler_fn(token_embeddings = x)
        x = self.cls_head(x)

        return x 

class TransfomerClassifier():
    def __init__(self, emb_dim, n_out, mode, patch_size, batch_size, epochs, n_channels=1, use_fnet=False, save_path=None):
        self.emb_dim = emb_dim
        self.n_out = n_out
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.patch_size = patch_size
        self.use_fnet=use_fnet
        self.n_channels= n_channels
        self.save_path = save_path

    def fit(self, X, y, X_va, y_va):

        self.clf = TransfomerBlock(self.patch_size, self.n_out, self.emb_dim, self.mode, self.n_channels, self.use_fnet).to("cuda")

        if len(X.shape) == 3:
            summary(self.clf, input_size=(2, X.shape[1], X.shape[2]),
            col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"], depth=5)
        else:
            summary(self.clf, input_size=(2, 2, X.shape[2], X.shape[3]),
            col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"], depth=5)

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
            total_time = end - start
            
            self.clf.eval()
            val_pred = []
            val_labels = []
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

            print("Epoch %d: " %epoch)
            print("BAS Score on Validation Data (Training Time %f): " %total_time, ba_score(val_labels, val_pred))
        if self.save_path is not None:
            torch.save(self.clf.state_dict(), self.save_path)
        return self

    def predict(self, X, y):
        
        test_pred = []

        X_te = torch.tensor(X).to("cuda")
        y_te = torch.tensor(y).to("cuda")
        
        dataset_test = torch.utils.data.DataLoader(
            list(zip(X_te, y_te)), shuffle=False, batch_size=self.batch_size
        )

        self.clf.eval()
        
        with torch.no_grad():
            for batch in dataset_test:
                x_v, _ = batch
    
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    outputs = self.clf(x_v)
    
                    pred = torch.max(outputs, 1)[1]
    
                    test_pred.extend(pred.cpu().numpy())

        test_pred = np.asarray(test_pred)

        return test_pred


