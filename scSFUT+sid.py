import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from my_model.mlp_cls_sid import MLP
from my_model.trans_enc_cls import PoolingLayer, TransformerEncoder
from my_model.util import setup_seed, count_labels, FocalLoss
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score
import copy
import argparse
import os

# Ignore UndefinedMetricWarning when cal precision
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Train with', device)

parser = argparse.ArgumentParser()
#Global Params
parser.add_argument("--dataset", type=str, default='Baron', help='Name of data for train, val and test.')
parser.add_argument("--batch_size", type=int, default=64, help='Number of batch.')
parser.add_argument("--num_epoch", type=int, default=25, help='Number of epochs.')
parser.add_argument("--num_fold", type=int, default=5, help='Number of fold.')
parser.add_argument("--seed", type=int, default=2025, help='Random seed.')
parser.add_argument("--learning_rate", type=float, default=0.0005, help='Learning rate.')
parser.add_argument("--scheduler_type", type=bool, default=True, help='CosineAnnealingLR exists or not.')
parser.add_argument("--w_recloss", type=float, default=0.5, help='Weight of recloss.')
parser.add_argument("--w_clsloss", type=float, default=1.5, help='Weight of cls loss(FocalLoss).')
# Params of MLP
parser.add_argument("--hidden_size_1", type=int, default=512, help='Size of Linear 1.')
parser.add_argument("--hidden_size_2", type=int, default=64, help='Size of Linear 2.')
parser.add_argument("--dropout", type=float, default=0.1, help='Dropout rate of classfier.')
#Params of Transformer
parser.add_argument("--token_dim", type=int, default=64, help='Dimension of token.')
parser.add_argument("--conv_dim", type=int, default=128, help='Dimension of conv block.')
parser.add_argument("--layer1", type=int, default=1, help='Layer num of Transformer encoder.')
parser.add_argument("--layer2", type=int, default=3, help='Layer num of Transformer decoder.')
parser.add_argument("--mask_percentage", type=float, default=0.4, help='Mask percentage of non_zero token, and the mask percentage of zero token is 1/10 of this num.')
parser.add_argument("--num_mulhead", type=int, default=8, help='Number of head in Transformer.')
parser.add_argument("--data_dir", type=str, default='results/preprocessed_samples', help='Directory for 5-fold npz data')
parser.add_argument("--sid_dir", type=str, default='results/sid_indices', help='Directory for sid features npz')
args = parser.parse_args()

DATASET_NAME = args.dataset
BATCH_SIZE = args.batch_size

EPOCH = 15
FOLD = args.num_fold
RANDOM_SEED = args.seed
LR = args.learning_rate
IF_COS = args.scheduler_type
W_REC = args.w_recloss
W_CLS = args.w_clsloss
HIDDEN_SIZE_1 = args.hidden_size_1
HIDDEN_SIZE_2 = args.hidden_size_2
DROPOUT = args.dropout
TOKEN_DIM = args.token_dim
CONV_DIM = args.conv_dim
NUM_LAYER1 = args.layer1
NUM_LAYER2 = args.layer2
M_PERCENTAGE = args.mask_percentage
NUM_HEAD = args.num_mulhead
DATA_DIR = args.data_dir
SID_DIR = args.sid_dir

def preprocess_features(features):
    """
    对输入特征进行归一化到1e4并log处理
    features: np.ndarray, shape (N, G)
    return: np.ndarray, shape (N, G)
    """
    # 归一化到1e4
    # 先对每个样本的总和归一化
    sums = features.sum(axis=1, keepdims=True)
    # 防止除以0
    sums[sums == 0] = 1
    normed = features / sums * 1e4
    # log1p处理
    loged = np.log1p(normed)
    return loged

class FoldDataset(Dataset):
    def __init__(self, data, labels, sid_features):
        self.data = data
        self.labels = labels
        self.sid_features = sid_features

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]).double(),
            torch.tensor(self.labels[idx]).long(),
            torch.from_numpy(self.sid_features[idx]).long()
        )

def seq2token(features, embedding_dim):
    """
    features: np.ndarray, shape (num_samples, num_features)
    embedding_dim: int
    Returns: np.ndarray, shape (num_samples, num_tokens, embedding_dim)
    """
    num_samples, num_features = features.shape
    remaining_features = num_features % embedding_dim
    if remaining_features != 0:
        padding_size = embedding_dim - remaining_features
        features_padded = np.concatenate([features, np.zeros((num_samples, padding_size))], axis=1)
    else:
        features_padded = features
    num_tokens = features_padded.shape[1] // embedding_dim
    grouped_features = features_padded.reshape(num_samples, num_tokens, embedding_dim)
    return grouped_features

def load_fold_data(fold_idx):
    # 文件路径
    train_data_path = os.path.join(DATA_DIR, DATASET_NAME, f"train_fold{fold_idx}_raw.npz")
    eval_data_path = os.path.join(DATA_DIR, DATASET_NAME, f"eval_fold{fold_idx}_raw.npz")
    train_sid_path = os.path.join(SID_DIR, DATASET_NAME, f"train_sid_fold{fold_idx}.npz")
    eval_sid_path = os.path.join(SID_DIR, DATASET_NAME, f"eval_sid_fold{fold_idx}.npz")

    # 读取train
    train_npz = np.load(train_data_path, allow_pickle=True)
    train_features = train_npz['X']  # shape: (N, G)
    train_labels = train_npz['y']  # shape: (N,)
    # 读取eval
    eval_npz = np.load(eval_data_path, allow_pickle=True)
    eval_features = eval_npz['X']
    eval_labels = eval_npz['y']
    # 读取sid
    train_sid = np.load(train_sid_path, allow_pickle=True)['sid']  # shape: (N, ?)
    eval_sid = np.load(eval_sid_path, allow_pickle=True)['sid']

    # 特征预处理
    train_features = preprocess_features(train_features)
    eval_features = preprocess_features(eval_features)

    # seq2token
    train_tokens = seq2token(train_features, TOKEN_DIM)
    eval_tokens = seq2token(eval_features, TOKEN_DIM)

    return train_tokens, train_labels, train_sid, eval_tokens, eval_labels, eval_sid

def train_and_eval():
    last_f1s = []
    last_accuracies = []
    last_precisions = []
    setup_seed(RANDOM_SEED)
    for fold in range(1, FOLD + 1):
        print(f"\n=== Fold {fold}/{FOLD} ===")
        # 读取数据
        train_data, train_labels, train_sid, eval_data, eval_labels, eval_sid = load_fold_data(fold)
        print(f"Train data shape: {train_data.shape}, Eval data shape: {eval_data.shape}")
        
        # Convert string labels to integer indices
        all_labels = np.concatenate([train_labels, eval_labels])
        unique_labels = np.unique(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Map string labels to integers
        train_labels_int = np.array([label_to_idx[label] for label in train_labels])
        eval_labels_int = np.array([label_to_idx[label] for label in eval_labels])
        
        NUM_CLASS = len(unique_labels)
        print(f"Number of classes: {NUM_CLASS}")
        print(f"Class labels: {unique_labels}")
        
        # Use integer labels for the rest of the training
        train_labels = train_labels_int
        eval_labels = eval_labels_int
        _, token_num, _ = train_data.shape

        train_dataset = FoldDataset(train_data, train_labels, train_sid)
        eval_dataset = FoldDataset(eval_data, eval_labels, eval_sid)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        # 初始化模型
        transformer_model = TransformerEncoder(
            seq_length=token_num,
            token_dim=TOKEN_DIM,
            conv_emb_dim=CONV_DIM,
            num_layers_1=NUM_LAYER1,
            num_layers_2=NUM_LAYER2,
            num_heads=NUM_HEAD,
            mask_percentage=M_PERCENTAGE
        ).double().to(device)
        classification_model = MLP(
            input_dim=token_num,
            hidden_dim1=HIDDEN_SIZE_1,
            hidden_dim2=HIDDEN_SIZE_2,
            num_classes=NUM_CLASS,
            dropout=DROPOUT,
            cat_dims=[40, 40, 40, 40, 40, 40],
            emb_dims=[28,28,28,28,28,28]
            # emb_dims = [4,4,4,4,4,4]
        ).double().to(device)

        criterion = FocalLoss(gamma=0)
        optimizer = torch.optim.Adam(
            list(transformer_model.parameters()) + list(classification_model.parameters()),
            lr=LR, weight_decay=1e-4
        )

        for epoch in tqdm(range(EPOCH), desc=f'Fold {fold}/{FOLD}'):
            transformer_model.train()
            classification_model.train()
            for data_batch, label_batch, sid_batch in train_loader:
                data_batch, label_batch, sid_batch = data_batch.to(device), label_batch.to(device), sid_batch.to(device)
                optimizer.zero_grad()
                transformer_output, rec_loss = transformer_model(data_batch)
                predictions = classification_model(transformer_output, sid_batch)
                loss = W_CLS * criterion(predictions, label_batch) + W_REC * rec_loss
                loss.backward()
                optimizer.step()

            # 验证
            transformer_model.eval()
            classification_model.eval()
            with torch.no_grad():
                all_val_predictions = []
                all_val_labels = []
                for val_data_batch, val_label_batch, val_sid_batch in eval_loader:
                    val_data_batch, val_label_batch, val_sid_batch = val_data_batch.to(device), val_label_batch.to(device), val_sid_batch.to(device)
                    val_transformer_output, _ = transformer_model(val_data_batch)
                    val_predictions = classification_model(val_transformer_output, val_sid_batch)
                    all_val_predictions.append(val_predictions.cpu().numpy())
                    all_val_labels.append(val_label_batch.cpu().numpy())
                all_val_predictions = np.concatenate(all_val_predictions)
                all_val_labels = np.concatenate(all_val_labels)
                val_pred_classes = np.argmax(all_val_predictions, axis=1)
                val_accuracy = accuracy_score(all_val_labels, val_pred_classes)
                val_f1 = f1_score(all_val_labels, val_pred_classes, average='macro')
                val_precision_final = precision_score(all_val_labels, val_pred_classes, average='macro')
                print(f"Epoch {epoch + 1}/{EPOCH}, Fold {fold}/{FOLD} - Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}, Val Precision_final Score: {val_precision_final:.4f}")

                if epoch == EPOCH - 1:
                    last_f1s.append(val_f1)
                    last_accuracies.append(val_accuracy)
                    last_precisions.append(val_precision_final)

        # 保存模型
        save_dir = f"saved_models/{DATASET_NAME}/fold_{fold}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(transformer_model.state_dict(), os.path.join(save_dir, "transformer_model.pth"))
        torch.save(classification_model.state_dict(), os.path.join(save_dir, "classification_model.pth"))
        print(f"Saved models for fold {fold} to {save_dir}")

    print(f"\nFinal (15th epoch) F1 across {FOLD} folds: {np.mean(last_f1s):.4f} ± {np.std(last_f1s):.4f}")
    print(f"Final (15th epoch) Accuracy across {FOLD} folds: {np.mean(last_accuracies):.4f} ± {np.std(last_accuracies):.4f}")
    print(f"Final (15th epoch) Precision across {FOLD} folds: {np.mean(last_precisions):.4f} ± {np.std(last_precisions):.4f}")

if __name__ == "__main__":
    setup_seed(RANDOM_SEED)
    train_and_eval()

