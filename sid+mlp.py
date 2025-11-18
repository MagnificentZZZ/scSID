import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder
from rqvae_model import *
from rqvae_tools import *
import scanpy as sc
import os
import pickle
import random

import torch
import numpy as np
import os

dataset = 'immune33k'
input_dim = 1000  
num_embeddings = 40
embedding_dim = 28
num_layers = 6
batch_size = 64
if dataset in ['kidney']:  
    rqvae_epochs = 150
elif dataset in ['mat']:
    rqvae_epochs = 120
elif dataset in ['immune33k']:
    rqvae_epochs = 50
elif dataset in ['Baron']:
    rqvae_epochs = 80
else: 
    rqvae_epochs = 20


init_all(2025)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_rqvae(model, train_loader, num_epochs, device, learning_rate=1e-3, fold=0):
    """训练RQVAE模型"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        rec_loss = 0
        quantization_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Fold {fold+1} - RQVAE Epoch {epoch+1}/{num_epochs}'):
            x = batch[0].to(device)
            outputs = model(x)
            loss = outputs['recon_loss'] + outputs['quantization_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss += outputs['recon_loss'].item()
            quantization_loss += outputs['quantization_loss'].item()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_rec_loss = rec_loss / len(train_loader)
        avg_quantization_loss = quantization_loss / len(train_loader)
        
        # if epoch % 5 == 0:  # 每5个epoch打印一次
        #     print(f'Fold {fold+1} - Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        #     print(f'Fold {fold+1} - Epoch {epoch+1}, Average Rec Loss: {avg_rec_loss:.4f}')
        #     print(f'Fold {fold+1} - Epoch {epoch+1}, Average Quan Loss: {avg_quantization_loss:.4f}')
        #     entropies = compute_entropy_from_loader(model, train_loader, device)
        #     print(f'Fold {fold+1} - Epoch {epoch+1}, Entropies: {entropies}')
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Fold {fold+1} - Epoch {epoch+1}')
            print(f'  Avg Total Loss: {avg_loss:.4f}')
            print(f'  Avg Recon Loss: {avg_rec_loss:.4f}')
            print(f'  Avg Quant Loss: {avg_quantization_loss:.4f}')
            
            # === 新增 ===
            metrics = compute_tokenizer_metrics(model, train_loader, device)
            print(f'  Reconstruction Loss (check): {metrics["reconstruction_loss"]:.4f}')
            print(f'  Codebook Utilization per layer: {[round(u, 4) for u in metrics["utilization"]]}')
            print(f'  Token Entropy per layer: {[round(e, 4) for e in metrics["entropy"]]}')
            # === 在最后一个epoch保存指标结果 ===
            if epoch == num_epochs - 1:
                os.makedirs(f'results/tokenizer_metrics/{dataset}', exist_ok=True)
                save_path = f'results/tokenizer_metrics/{dataset}/fold{fold+1}_final_metrics.pkl'
                torch.save(metrics, save_path)
                print(f'✅ Tokenizer metrics saved to: {save_path}')


def get_codebook_indices(model, data_loader, device):
    """获取数据的码本索引"""
    model.eval()
    all_indices = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            indices_list = model.get_codebook_indices(x)  # List[layer] -> Tensor[b]
            # 将所有层的索引拼接成一个向量
            batch_indices = torch.stack(indices_list, dim=1)  # [batch_size, num_layers]
            all_indices.append(batch_indices.cpu())
    
    return torch.cat(all_indices, dim=0)

class CodebookClassifier(nn.Module):
    """基于码本索引的分类器，使用embedding层+MLP"""
    def __init__(self, num_layers, num_embeddings, embedding_dim, num_classes):
        super(CodebookClassifier, self).__init__()
        self.num_layers = num_layers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 为每一层创建embedding层
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) 
            for _ in range(num_layers)
        ])
    
        # 计算输入维度：所有层的embedding拼接
        total_embedding_dim = num_layers * embedding_dim
        
        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(total_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, indices):
        # indices shape: [batch_size, num_layers]
        batch_size = indices.size(0)
        
        # 对每一层的索引进行embedding
        embeddings_list = []
        for i in range(self.num_layers):
            layer_indices = indices[:, i]  # [batch_size]
            layer_embeddings = self.embeddings[i](layer_indices)  # [batch_size, embedding_dim]
            embeddings_list.append(layer_embeddings)
        
        # 拼接所有层的embeddings
        combined_embeddings = torch.cat(embeddings_list, dim=1)  # [batch_size, num_layers * embedding_dim]
        
        # 通过MLP分类器
        output = self.classifier(combined_embeddings)
        return output

def train_classifier(indices_train, labels_train, indices_val, labels_val, 
                    num_classes, num_layers, num_embeddings, embedding_dim, fold=0):
    """训练基于码本索引的分类器，使用早停机制"""
    # 创建分类模型
    classifier = CodebookClassifier(
        num_layers=num_layers,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    ).to(device)
    
    # 准备数据
    train_dataset = TensorDataset(
        indices_train.long(),
        labels_train.long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 准备验证数据
    val_dataset = TensorDataset(
        indices_val.long(),
        labels_val.long()
    )
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 训练分类器
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    best_train_loss = float('inf')
    if dataset in ['mat','kidney']:
       patience = 6
    elif dataset in ['zheng68k']:
       patience = 3 
    else:
       patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        classifier.train()
        train_loss = 0
        
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 早停检查
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            patience_counter = 0
            best_model_state = classifier.state_dict().copy()
        else:
            patience_counter += 1
            
        if epoch % 2 == 0:
            # 计算验证集性能
            classifier.eval()
            val_accuracy, val_f1, val_precision = evaluate_classifier(classifier, val_loader, device)
            print(f'Fold {fold+1} - Classifier Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}')
            classifier.train()  # 切换回训练模式
            
        # 如果连续patience轮没有改善，停止训练
        if patience_counter >= patience:
            print(f'Fold {fold+1} - Early stopping at epoch {epoch+1}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
    
    return classifier

def evaluate_classifier(classifier, data_loader, device):
    """评估分类器"""
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            outputs = classifier(x)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1, precision

def select_highly_variable_genes_and_norm(h5ad_file_path, n_top_genes=1000):
    """
    选择高变基因，直接对高变基因筛选后的1000个基因做norm10000+log1p。
    返回：df_hvg, adata
    """
    try:
        adata = sc.read_h5ad(h5ad_file_path)
        # 识别高度可变基因
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3',
            subset=False
        )
        highly_variable_genes = adata.var_names[adata.var['highly_variable']].tolist()
        # 只保留高变基因
        adata_hvg = adata[:, highly_variable_genes].copy()
        # 直接对高变基因做norm10000和log1p
        sc.pp.normalize_total(adata_hvg, target_sum=1e4)
        sc.pp.log1p(adata_hvg)
        # 构建表达矩阵
        df_hvg = pd.DataFrame(
            adata_hvg.X.toarray() if hasattr(adata_hvg.X, "toarray") else adata_hvg.X,
            columns=highly_variable_genes,
            index=adata_hvg.obs_names
        )
        df_hvg['vector'] = list(df_hvg.values)
        if dataset in ['mat','kidney']:
            df_hvg['label'] = adata_hvg.obs['cell_ontology_class']
        elif dataset in ['immune33k']:
            df_hvg['label'] = adata_hvg.obs['final_annotation']
        elif dataset in ['Baron']:
            df_hvg['label'] = adata_hvg.obs['celltype']
        else:
            df_hvg['label'] = adata_hvg.obs['cell_type']
        print(f"成功选择了{n_top_genes}个高变基因，并创建了{df_hvg.shape[0]}×{df_hvg.shape[1]}的表达矩阵。")
        return df_hvg, adata_hvg
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None, None

def run_cross_validation():
    """
    运行五折交叉验证，保存预处理样本和SID到单独文件夹。
    不再进行参数搜索。
    保存到本地的五折划分后的是原始样本，而不是高变基因筛选后的。
    """
    # 创建结果保存目录
    os.makedirs('results', exist_ok=True)
    # 新建文件夹保存五折样本和SID
    preproc_dir = f'results/preprocessed_samples/{dataset}'
    sid_dir = f'results/sid_indices/{dataset}'
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(sid_dir, exist_ok=True)
    # 加载数据
    h5ad_file = f"cls_data/{dataset}.h5ad"
    hvg_df, adata_hvg = select_highly_variable_genes_and_norm(h5ad_file, n_top_genes=input_dim)
    if hvg_df is None:
        print("数据加载失败！")
        return

    # 读取原始数据（未筛选高变基因，未归一化/对数化）
    adata_raw = sc.read_h5ad(h5ad_file)
    # 获取原始表达矩阵和标签
    X_raw = adata_raw.X.toarray() if hasattr(adata_raw.X, "toarray") else adata_raw.X
    if dataset in ['mat','kidney']:
        y_raw = adata_raw.obs['cell_ontology_class'].values
    elif dataset in ['immune33k']:
        y_raw = adata_raw.obs['final_annotation'].values
    elif dataset in ['Baron']:
        y_raw = adata_raw.obs['celltype'].values
    else:
        y_raw = adata_raw.obs['cell_type'].values

    # 用高变基因筛选后的数据做RQVAE训练和评估
    embeddings = torch.FloatTensor(np.stack(hvg_df['vector'].values))
    labels = hvg_df['label'].values
    # 标签编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"数据集大小: {len(embeddings)}")
    print(f"类别数: {num_classes}")
    print(f"类别: {label_encoder.classes_}")

    # 五折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    # 存储结果
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(embeddings, encoded_labels)):
        print(f"\n=== 开始第 {fold + 1} 折交叉验证 ===")
        # 为每一折设置不同的随机种子，保证每折完全可复现
        # 划分数据
        train_embeddings = embeddings[train_idx]
        val_embeddings = embeddings[val_idx]
        train_labels = torch.LongTensor(encoded_labels[train_idx])
        val_labels = torch.LongTensor(encoded_labels[val_idx])
        print(f"训练集大小: {len(train_embeddings)}, 验证集大小: {len(val_embeddings)}")

        # 保存五折划分的原始样本（未筛选高变基因，未归一化/对数化）
        np.savez_compressed(f"{preproc_dir}/train_fold{fold+1}_raw.npz",
                            X=X_raw[train_idx],
                            y=y_raw[train_idx])
        np.savez_compressed(f"{preproc_dir}/eval_fold{fold+1}_raw.npz",
                            X=X_raw[val_idx],
                            y=y_raw[val_idx])

        # 第一阶段：训练RQVAE
        print("第一阶段：训练RQVAE...")
        # 创建RQVAE模型
        rqvae_model = RQVAE(
            input_dim=input_dim,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        ).to(device)
        # 初始化码本
        sample_size = min(2000, len(train_embeddings))
        # 这里保证采样顺序可复现
        indices = torch.randperm(len(train_embeddings))[:sample_size]
        sampled_embeddings = train_embeddings[indices]
        rqvae_model.initialize_codebook(sampled_embeddings.to(device))
        # 训练RQVAE
        train_dataset = TensorDataset(train_embeddings)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        train_rqvae(rqvae_model, train_loader, rqvae_epochs, device, fold=fold)
        # 保存RQVAE模型
        os.makedirs(f'results/{dataset}', exist_ok=True)
        torch.save(rqvae_model.state_dict(), f'results/{dataset}/rqvae_fold_{fold+1}.pth')
        # 第二阶段：获取码本索引并训练分类器
        print("第二阶段：获取码本索引...")
        # 获取训练集和验证集的码本索引
        train_loader = DataLoader(TensorDataset(train_embeddings), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(val_embeddings), batch_size=batch_size, shuffle=False)
        train_indices = get_codebook_indices(rqvae_model, train_loader, device)
        val_indices = get_codebook_indices(rqvae_model, val_loader, device)
        # 保存SID（码本索引），改为.npz格式
        np.savez_compressed(f"{sid_dir}/train_sid_fold{fold+1}.npz", sid=train_indices.numpy())
        np.savez_compressed(f"{sid_dir}/eval_sid_fold{fold+1}.npz", sid=val_indices.numpy())
        print("训练分类器...")
        # 训练分类器
        classifier = train_classifier(
            train_indices, train_labels, val_indices, val_labels,
            num_classes, num_layers, num_embeddings, embedding_dim, fold=fold
        )
        # 评估分类器
        val_dataset = TensorDataset(
            val_indices.long(),
            val_labels.long()
        )
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        accuracy, f1, precision = evaluate_classifier(classifier, val_loader, device)
        # 保存结果
        fold_result = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision
        }
        fold_results.append(fold_result)
        print(f"第 {fold + 1} 折结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  精确率: {precision:.4f}")
        # 保存分类器，保存到dataset子文件夹下
        torch.save(classifier.state_dict(), f'results/{dataset}/classifier_fold_{fold+1}.pth')
    # 计算所有折的统计结果
    accuracies = [result['accuracy'] for result in fold_results]
    f1_scores = [result['f1'] for result in fold_results]
    precisions = [result['precision'] for result in fold_results]
    # 计算均值和标准差
    results_summary = {
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies)
        },
        'f1': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores)
        },
        'precision': {
            'mean': np.mean(precisions),
            'std': np.std(precisions)
        }
    }
    # 打印最终结果
    print("\n=== 五折交叉验证最终结果 ===")
    print(f"准确率: {results_summary['accuracy']['mean']:.4f} ± {results_summary['accuracy']['std']:.4f}")
    print(f"F1分数: {results_summary['f1']['mean']:.4f} ± {results_summary['f1']['std']:.4f}")
    print(f"精确率: {results_summary['precision']['mean']:.4f} ± {results_summary['precision']['std']:.4f}")
    rqvae_matrix = summarize_tokenizer_metrics(metrics_dir=f'results/tokenizer_metrics/{dataset}', num_folds=5)
    # 保存详细结果
    with open('results/cross_validation_results.pkl', 'wb') as f:
        pickle.dump({
            'fold_results': fold_results,
            'summary': results_summary,
            'label_encoder': label_encoder,
            'rqvae_matrix':rqvae_matrix
        }, f)
    # 保存结果到CSV
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('results/cross_validation_results.csv', index=False)
    # 保存汇总结果到CSV
    summary_df = pd.DataFrame([{
        'accuracy_mean': results_summary['accuracy']['mean'],
        'accuracy_std': results_summary['accuracy']['std'],
        'f1_mean': results_summary['f1']['mean'],
        'f1_std': results_summary['f1']['std'],
        'precision_mean': results_summary['precision']['mean'],
        'precision_std': results_summary['precision']['std']
    }])
    summary_df.to_csv('results/summary_results.csv', index=False)
    return results_summary

if __name__ == "__main__":
    # 主入口再次设置随机种子，保证所有全局操作可复现
    results = run_cross_validation()
