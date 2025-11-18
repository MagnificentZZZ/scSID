# import torch
# import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, dropout, 
#                  cat_dims = None, emb_dims = None):
#         """
#         扩展的MLP模型，支持数值特征和类别特征
        
#         参数:
#         input_dim (int): 原始数值输入特征维度
#         hidden_dim1 (int): 第一个隐藏层维度
#         hidden_dim2 (int): 第二个隐藏层维度
#         num_classes (int): 输出类别数量
#         dropout (float): Dropout概率
#         cat_dims (list): 每个类别特征的唯一值数量列表
#         emb_dims (list): 每个类别特征的嵌入维度列表
#         """
#         super(MLP, self).__init__()
        
#         # 原始数值特征处理
#         self.input_dim = input_dim
        
#         # 类别特征的embedding layers
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(cat_dim, emb_dim) 
#             for cat_dim, emb_dim in zip(cat_dims, emb_dims)
#         ])
        
#         # 计算嵌入层输出的总维度
#         total_emb_dim = sum(emb_dims)
        
#         # 拼接后的输入维度
#         combined_dim = input_dim + total_emb_dim
        
#         # 全连接层
#         self.fc1 = nn.Linear(combined_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc3 = nn.Linear(hidden_dim2, num_classes)

#     def forward(self, x, cat_features):
#         """
#         前向传播方法
        
#         参数:
#         x (Tensor): 数值输入特征
#         cat_features (list): 类别特征列表，每个元素是一个Tensor
#         """
#         # 处理类别特征
#         embeddings = []
#          # 为每个类别特征应用对应的embedding层
#         cat_embedded_list = [
#             self.embeddings[i](cat_features[:, i].long())
#             for i in range(cat_features.shape[1])
#         ]
        
#         # 拼接所有嵌入结果
#         if cat_embedded_list:
#             cat_embedded = torch.cat(cat_embedded_list, dim=1)
#             combined = torch.cat([x, cat_embedded], dim=1)
#         else:
#             combined = x
        
#         # 原始MLP处理流程
#         x = self.fc1(combined)
#         x = self.relu(x)
        
#         x = self.fc2(x)
#         x = self.dropout(x)
        
#         x = self.fc3(x)
        
#         return x


import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, dropout, 
                 cat_dims=None, emb_dims=None):
        """
        扩展的MLP模型，支持数值特征和类别特征
        
        参数:
        input_dim (int): 原始数值输入特征维度
        hidden_dim1 (int): 第一个隐藏层维度
        hidden_dim2 (int): 第二个隐藏层维度
        num_classes (int): 输出类别数量
        dropout (float): Dropout概率
        cat_dims (list): 每个类别特征的唯一值数量列表
        emb_dims (list): 每个类别特征的嵌入维度列表
        """
        super(MLP, self).__init__()
        
        self.use_cat_features = cat_dims is not None and emb_dims is not None
        
        if self.use_cat_features:
            # 类别特征的embedding layers
            self.embeddings = nn.ModuleList([
                nn.Embedding(cat_dim, emb_dim) 
                for cat_dim, emb_dim in zip(cat_dims, emb_dims)
            ])
            # 计算嵌入层输出的总维度
            total_emb_dim = sum(emb_dims)
            # 拼接后的输入维度
            combined_dim = input_dim + total_emb_dim
        else:
            self.embeddings = None
            combined_dim = input_dim
        
        # 全连接层
        self.fc1 = nn.Linear(combined_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, cat_features=None):
        """
        前向传播方法
        
        参数:
        x (Tensor): 数值输入特征
        cat_features (list): 类别特征列表，每个元素是一个Tensor
        """
        if self.use_cat_features and cat_features is not None:
            # 处理类别特征
            # 为每个类别特征应用对应的embedding层
            cat_embedded_list = [
                self.embeddings[i](cat_features[:, i].long())
                for i in range(cat_features.shape[1])
            ]
            
            # 拼接所有嵌入结果
            if cat_embedded_list:
                cat_embedded = torch.cat(cat_embedded_list, dim=1)
                combined = torch.cat([x, cat_embedded], dim=1)
            else:
                combined = x
        else:
            combined = x
        
        # 原始MLP处理流程
        x = self.fc1(combined)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x