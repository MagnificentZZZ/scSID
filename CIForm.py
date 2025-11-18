import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import math
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
import scipy.sparse

s = 1024  # the length of a sub-vector

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_and_split(train_features, train_labels, val_features, val_labels, topgenes, gap):
    """
    Corrected preprocessing pipeline that prevents data leakage.
    1. HVGs are selected based *only* on the training data.
    2. The same set of HVGs is then applied to the validation data.
    """
    # 1. Create AnnData objects for train and validation sets
    train_adata = sc.AnnData(train_features)
    train_adata.obs['labels'] = train_labels

    val_adata = sc.AnnData(val_features)
    val_adata.obs['labels'] = val_labels
    
    # Ensure both datasets have the same gene names before selection
    val_adata.var_names = train_adata.var_names

    # 2. Preprocess the TRAINING DATA to find highly variable genes
    # All learning steps are performed ONLY on train_adata
    sc.pp.normalize_total(train_adata, target_sum=1e4)
    sc.pp.log1p(train_adata)
    
    # Adjust topgenes if the number of genes in the training data is smaller
    if train_adata.shape[1] < topgenes:
        topgenes = train_adata.shape[1]

    # Learn HVGs from the training data
    sc.pp.highly_variable_genes(train_adata, n_top_genes=topgenes, flavor='seurat_v3')
    
    # Get the boolean mask of highly variable genes from the training data
    hvg_mask = train_adata.var['highly_variable']
    
    # Filter the training data to keep only these HVGs
    train_adata = train_adata[:, hvg_mask].copy()

    # 3. Preprocess the VALIDATION DATA separately
    sc.pp.normalize_total(val_adata, target_sum=1e4)
    sc.pp.log1p(val_adata)
    
    # Filter the validation data to keep THE SAME genes selected from the training data
    # This prevents data leakage and ensures feature consistency.
    val_adata = val_adata[:, hvg_mask].copy()

    # 4. Extract processed expression matrices
    X_train_processed = train_adata.X.toarray() if isinstance(train_adata.X, scipy.sparse.spmatrix) else train_adata.X
    X_val_processed = val_adata.X.toarray() if isinstance(val_adata.X, scipy.sparse.spmatrix) else val_adata.X

    # 5. Define the function to split features into sub-vectors and scale them
    def split_and_scale_subvectors(expression_matrix, gap_size):
        subvector_list = []
        for single_cell in expression_matrix:
            feature = []
            length = len(single_cell)
            for k in range(0, length, gap_size):
                if k + gap_size > length:
                    sub_vector = single_cell[k:]
                    padding = np.zeros(gap_size - len(sub_vector))
                    a = np.concatenate([sub_vector, padding])
                else:
                    a = single_cell[k:k + gap_size]
                # Scale each sub-vector. This is local and doesn't leak info between cells.
                a = preprocessing.scale(a)
                feature.append(a)
            subvector_list.append(np.array(feature))
        return np.array(subvector_list)

    # Apply the splitting and scaling to each dataset independently
    X_train = split_and_scale_subvectors(X_train_processed, gap)
    X_val = split_and_scale_subvectors(X_val_processed, gap)
    
    y_train = train_adata.obs['labels'].values
    y_val = val_adata.obs['labels'].values
    
    return X_train, y_train, X_val, y_val

def load_fold_data(fold_idx, data_dir, dataset_name, sid_dir, topgenes, gap):
    """
    Loads raw data for a specific fold from .npz files and preprocesses it.
    """
    # Define file paths
    train_data_path = os.path.join(data_dir, dataset_name, f"train_fold{fold_idx}_raw.npz")
    eval_data_path = os.path.join(data_dir, dataset_name, f"eval_fold{fold_idx}_raw.npz")
    # Assuming a separate test file exists, otherwise eval can be split or used as test
    
    train_sid_path = os.path.join(sid_dir, dataset_name, f"train_sid_fold{fold_idx}.npz")
    eval_sid_path = os.path.join(sid_dir, dataset_name, f"eval_sid_fold{fold_idx}.npz")

    # Load raw feature and label data
    train_npz = np.load(train_data_path, allow_pickle=True)
    train_features, train_labels = train_npz['X'], train_npz['y']
    
    eval_npz = np.load(eval_data_path, allow_pickle=True)
    eval_features, eval_labels = eval_npz['X'], eval_npz['y']

    # Load side information
    sid_train = np.load(train_sid_path, allow_pickle=True)['sid']
    sid_val = np.load(eval_sid_path, allow_pickle=True)['sid']

    # Perform the full preprocessing and splitting pipeline
    X_train, y_train, X_val, y_val = preprocess_and_split(
        train_features, train_labels, eval_features, eval_labels, topgenes, gap
    )
    
    # Encode labels to numerical indices
    all_labels = np.unique(np.concatenate([y_train, y_val]))
    label_mapping = {label: i for i, label in enumerate(all_labels)}
    
    y_train = np.array([label_mapping[lbl] for lbl in y_train])
    y_val = np.array([label_mapping[lbl] for lbl in y_val])

    return X_train, y_train, sid_train, X_val, y_val, sid_val, all_labels

class myDataSet(Dataset):
    def __init__(self, data, label, sid=None):
        self.data = data
        self.label = label
        self.sid = sid
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index])
        label = torch.tensor(self.label[index], dtype=torch.long)
        
        if self.sid is not None:
            sid = torch.from_numpy(self.sid[index]).long()
            return data, sid, label
        else:
            return data, label

##Positional Encoder Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class CIForm(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1, 
                 use_sid=False, sid_dims=None, sid_emb_dim=4):
        super().__init__()
        self.use_sid = use_sid
        self.d_model = d_model

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=1024, nhead=nhead, dropout=dropout
        )
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        main_feature_dim = d_model
        
        if self.use_sid:
            self.sid_embeddings = nn.ModuleList([
                nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in sid_dims
            ])
            total_sid_emb_dim = sum([emb_dim for _, emb_dim in sid_dims])
            classifier_input_dim = d_model + total_sid_emb_dim
        else:
            classifier_input_dim = d_model

        self.pred_layer = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, mels, sid=None):
        out = mels.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        out = out.mean(dim=1)
        
        if self.use_sid and sid is not None:
            sid_embs = []
            for i, emb_layer in enumerate(self.sid_embeddings):
                sid_embs.append(emb_layer(sid[:, i]))
            
            sid_out = torch.cat(sid_embs, dim=1)
            out = torch.cat([out, sid_out], dim=1)
            
        out = self.pred_layer(out)
        return out

def main(s, dataset_name, seed, use_sid=False, T=False):
    # --- Model & Training Parameters ---
    gap = s
    d_models = s
    heads = 16
    lr = 0.0001
    dp = 0.1
    n_epochs = 50  # Increased epochs for early stopping to be effective
    batch_size = 256
    
    # --- Early Stopping Parameters ---
    patience = 6  # Number of epochs to wait for improvement before stopping
    min_delta = 0.001 # Minimum change in val_loss to be considered an improvement

    # --- Data Paths ---
    DATA_DIR = f"results/preprocessed_samples/" # Specify your data directory
    SID_DIR = f"results/sid_indices/"   # Specify your sid directory
    if use_sid:
        MODEL_SAVE_DIR = f"saved_models/CIForm/sid/{dataset_name}" # Define a dedicated folder for saved models
    else:
        MODEL_SAVE_DIR = f"saved_models/CIForm/{dataset_name}" # Define a dedicated folder for saved models


    # --- Lists to store metrics across folds ---
    # (The original test lists are kept in case you add test evaluation later)
    all_test_accs = []
    all_test_pres = []
    all_test_f1s = []
    
    # Lists for validation metrics as requested
    all_val_accs = []
    all_val_f1s = []
    all_val_pres = []

    for fold in range(5):
        print(f"Processing Fold {fold + 1}")
        
        # Note: Assuming load_fold_data returns train and validation sets.
        # If it returns a test set, you would need to load that separately.
        X_train, y_train, sid_train, X_val, y_val, sid_val, cell_types = load_fold_data(
            fold + 1, DATA_DIR, dataset_name, SID_DIR, topgenes=3000, gap=1024
        )
        print(cell_types)
        
        num_classes = len(cell_types)
        
        model_params = {
            'input_dim': d_models, 'nhead': heads, 'd_model': d_models,
            'num_classes': num_classes, 'dropout': dp, 'use_sid': use_sid
        }
        
        if use_sid:
            token_dim = 4
            sid_dims = [(40, token_dim), (40, token_dim), (40, token_dim), (40, token_dim), (40, token_dim), (40, token_dim)] 
            model_params['sid_dims'] = sid_dims

        model = CIForm(**model_params)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay= 2* 1e-5)

        # --- Create Datasets and DataLoaders ---
        train_dataset = myDataSet(X_train, y_train, sid=sid_train if use_sid else None)
        val_dataset = myDataSet(X_val, y_val, sid=sid_val if use_sid else None)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # --- Initialize for Early Stopping ---
        best_val_loss = float('inf')
        epochs_no_improve = 0
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        best_model_path = os.path.join(MODEL_SAVE_DIR, f"model_fold_{fold + 1}.pth")

        # --- Training Loop with Early Stopping ---
        for epoch in range(n_epochs):
            model.train()
            train_loss, train_accs = [], []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
                labels = batch[-1]
                logits = model(*batch[:-1]) if use_sid else model(batch[0])
                
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(1)
                acc = (preds == labels).float().mean().item()
                train_loss.append(loss.item())
                train_accs.append(acc)

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            # --- Validation Step ---
            model.eval()
            val_losses, val_accs = [], []
            with torch.no_grad():
                for batch in val_loader:
                    labels = batch[-1]
                    logits = model(*batch[:-1]) if use_sid else model(batch[0])
                    
                    loss = criterion(logits, labels)
                    preds = logits.argmax(1)
                    acc = (preds == labels).float().mean().item()
                    val_losses.append(loss.item())
                    val_accs.append(acc)

            val_loss = sum(val_losses) / len(val_losses)
            val_acc = sum(val_accs) / len(val_accs)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # --- Early Stopping Check ---
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Validation loss decreased to {best_val_loss:.4f}. Saving model.")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        # --- Evaluate the best model for this fold on the validation set ---
        print(f"Loading best model from {best_model_path} for final validation.")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch[-1]
                logits = model(*batch[:-1]) if use_sid else model(batch[0])
                preds = logits.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        # Calculate metrics for the best model of the current fold
        fold_val_acc = accuracy_score(val_labels_list, val_preds)
        fold_val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        fold_val_pre = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        
        all_val_accs.append(fold_val_acc)
        all_val_f1s.append(fold_val_f1)
        all_val_pres.append(fold_val_pre)
        # --- Save predictions and labels for Fold 1 ---
        # if fold == 0:  # Only save for the first fold
        save_pred_dir = os.path.join(MODEL_SAVE_DIR, "predictions")
        os.makedirs(save_pred_dir, exist_ok=True)
        save_pred_path = os.path.join(save_pred_dir, f"{dataset_name}_fold{fold + 1}_predictions.npz")

        np.savez(
            save_pred_path,
            true_labels=np.array(val_labels_list),
            pred_labels=np.array(val_preds),
            class_names=np.array(cell_types)
        )
        print(f"Saved fold {fold + 1} predictions and labels to: {save_pred_path}")

        
        print(f"Fold {fold + 1} Best Validation - Acc: {fold_val_acc:.4f}, F1: {fold_val_f1:.4f}, Pre: {fold_val_pre:.4f}")
        print("-" * 50)

    # --- Calculate and Print Overall Validation Results ---
    mean_val_acc = np.mean(all_val_accs)
    std_val_acc = np.std(all_val_accs) # Standard deviation is the square root of variance
    mean_val_f1 = np.mean(all_val_f1s)
    std_val_f1 = np.std(all_val_f1s)
    mean_val_pre = np.mean(all_val_pres)
    std_val_pre = np.std(all_val_pres)

    print("\n" + "=" * 60)
    print("--- Overall 5-Fold Cross-Validation Results (on Validation Set) ---")
    print(f"Mean Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Mean Validation F1-Score: {mean_val_f1:.4f} ± {std_val_f1:.4f}")
    print(f"Mean Validation Precision: {mean_val_pre:.4f} ± {std_val_pre:.4f}")
    print("=" * 60)


same_seeds(2025)
main(s = 1024, dataset_name = 'mat', seed = 2025, use_sid = True)

