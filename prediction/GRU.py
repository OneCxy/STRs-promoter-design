# python /root/autodl-tmp/cxy/CODE/STR/prediction/GRU.py

import glob
import os
import random
import time
import sys

sys.path.append('/root/autodl-tmp/cxy/CODE/STR')

from prediction.NonLocalBlock1D import NonLocalBlock1D, MultiHeadAttention



from sklearn import metrics
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader

import re
import numpy as np
import pandas as pd
import torch
from torch import nn, functional
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from prediction.pytorchtools import EarlyStopping
import matplotlib.pyplot as plt


torch.nn.utils.clip_grad_norm_


class SAModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(SAModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.lambda_l2 = lambda_l2  # L2 正则化系数
        self.non_local_block = NonLocalBlock1D(hidden_size)
        self.pooling = nn.MaxPool1d(kernel_size=3)  # 使用平均池化作为例子
        self.bigru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))  # 定义权重
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):

        x = x.permute(0,2,1) # 调整输入维度顺序以适应CNN的输入格式
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.non_local_block(x)
        x = x.permute(0,2,1)  # 调整维度顺序以适应GRU的输入格式
        output, _ = self.bigru(x)  # BiGRU的输出包含正向和反向的隐藏状态
        output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=-1)  # 将正向和反向的隐藏状态拼接起来
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight  # 使用权重调整输出

        return output

class MultiModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(MultiModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.lambda_l2 = lambda_l2  # L2 正则化系数
        self.non_local_block = MultiHeadAttention(hidden_size,num_heads=4)
        self.pooling = nn.MaxPool1d(kernel_size=3)  # 使用平均池化作为例子
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.bigru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))  # 定义权重
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):

        x = x.permute(0,2,1) # 调整输入维度顺序以适应CNN的输入格式
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.permute(0, 2, 1)
        key = x
        value = x
        query = self.query_linear(x)
        query = query.unsqueeze(1)
        query = query.view(query.size(0), -1, self.hidden_size)  # 将 query 重新调整为三维张量
        x = self.non_local_block(query,key,value)
        output, _ = self.bigru(x)  # BiGRU的输出包含正向和反向的隐藏状态
        output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=-1)  # 将正向和反向的隐藏状态拼接起来
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight  # 使用权重调整输出

        return output

class MuGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(MuGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.lambda_l2 = lambda_l2  # L2 正则化系数
        self.non_local_block = MultiHeadAttention(hidden_size,num_heads=4)
        self.pooling = nn.MaxPool1d(kernel_size=3)  # 使用平均池化作为例子
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        # self.bigru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))  # 定义权重
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):

        x = x.permute(0,2,1) # 调整输入维度顺序以适应CNN的输入格式
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.permute(0, 2, 1)
        key = x
        value = x
        query = self.query_linear(x)
        query = query.unsqueeze(1)
        query = query.view(query.size(0), -1, self.hidden_size)  # 将 query 重新调整为三维张量
        x = self.non_local_block(query,key,value)

        # x = x.permute(0,2,1)  # 调整维度顺序以适应GRU的输入格式
        _, hidden = self.gru(x)
        output = hidden.squeeze(0)

        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight  # 使用权重调整输出

        return output


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.lambda_l2 = lambda_l2  # L2 正则化系数
        self.non_local_block = NonLocalBlock1D(hidden_size)
        # self.non_local_block = MultiHeadAttention(hidden_size,num_heads=4)
        self.pooling = nn.MaxPool1d(kernel_size=3)  # 使用平均池化作为例子
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        # self.bigru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))  # 定义权重
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):

        x = x.permute(0,2,1) # 调整输入维度顺序以适应CNN的输入格式
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.non_local_block(x)
        x = x.permute(0,2,1)  # 调整维度顺序以适应GRU的输入格式
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)
        hidden = self.fc1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        output = output * self.weight  # 使用权重调整输出
        # output, _ = self.bigru(x)  # BiGRU的输出包含正向和反向的隐藏状态
        # output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=-1)  # 将正向和反向的隐藏状态拼接起来
        # output = self.fc1(output)
        # output = self.relu(output)
        # output = self.dropout(output)
        # output = self.fc2(output)
        # output = output * self.weight  # 使用权重调整输出

        return output

class onlyGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, lambda_l2):
        super(onlyGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, dilation=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.lambda_l2 = lambda_l2  # L2 正则化系数
        self.pooling = nn.MaxPool1d(kernel_size=3)  # 使用平均池化作为例子
        # self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bigru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.Tensor([0.8]))  # 定义权重
        self.use_cuda = True if torch.cuda.is_available() else False

    def forward(self, x):
        # _, hidden = self.gru(x)
        # hidden = hidden.squeeze(0)
        # hidden = self.fc1(hidden)
        # hidden = self.relu(hidden)
        # hidden = self.dropout(hidden)
        # output = self.fc2(hidden)
        # output = output * self.weight  # 使用权重调整输出
        output, _ = self.bigru(x)  # BiGRU的输出包含正向和反向的隐藏状态
        output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=-1)  # 将正向和反向的隐藏状态拼接起来
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output * self.weight  # 使用权重调整输出

        return output

class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true):
        loss = torch.mean(self.weight * (y_true - y_pred)**2)
        return loss

class PREDICT():
    def __init__(self,run_name='predictecoli',dataset='/root/autodl-tmp/cxy/CODE/STR/data/ecoli_prediction.csv'):
        self.patience = 50
        self.val_acc_list = []
        self.save_path = '/root/autodl-tmp/cxy/CODE/STR/results/model/'
        self.dataset = dataset
        self.seq1, self.exp = self.data_load(dataset)
        self.seq = self.seq_onehot(self.seq1)
        self.train_idx, self.val_idx = self.split_by_activity_bins_8_2(seed=42)
        input_size = self.seq.shape[-1]
        self.input_size = input_size
        self.batch_size = 64
        self.hidden_size = 256
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.r = 11000
        self.use_gpu = True if torch.cuda.is_available() else False
        self.build_model()
        self.checkpoint_dir = '/root/autodl-tmp/cxy/CODE/STR/results/checkpoint/' + run_name + '/'
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)


    def string_to_array(self,my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    def one_hot_encode(self,my_array):
        label_encoder = LabelEncoder()
        label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))
        integer_encoded = label_encoder.transform(my_array)
        onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded

    def data_load(self, dataset):
        import pandas as pd
        import numpy as np
        import re

        # 1) 先尝试常见分隔符：逗号、tab、分号、空白
        for sep in [",", "\t", ";", r"\s+"]:
            try:
                df = pd.read_csv(dataset, header=None, sep=sep, engine="python")
                if df.shape[1] >= 2:
                    seq = df.iloc[:, 0].astype(str).tolist()
                    exp = df.iloc[:, 1].astype(float).values
                    expression = exp.reshape(-1, 1)
                    print(f"[data_load] sep={sep} shape={df.shape}")
                    return seq, expression
            except:
                pass

        # 2) 如果始终只有1列：从每行字符串里“抓最后一个数字”当标签
        df = pd.read_csv(dataset, header=None)
        col = df.iloc[:, 0].astype(str).tolist()

        seq, exp = [], []
        for s in col:
            s = s.strip()
            if not s:
                continue
            # 抓行末的数字（支持负号/小数）
            m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*$', s)
            if not m:
                continue
            y = float(m.group(1))
            dna = s[:m.start()].strip().replace(",", " ")
            dna = dna.split()[0]  # 取前面那段DNA
            seq.append(dna)
            exp.append(y)

        if len(seq) == 0:
            raise ValueError("data_load: 无法解析数据，请检查文件分隔符/内容。")

        expression = np.array(exp, dtype=float).reshape(-1, 1)
        print(f"[data_load] fallback-parse: {len(seq)} samples")
        return seq, expression

    def _bin_activity(self, y, min_bin=-6, max_bin=8):
        """
        把活性 y 分到 1 宽度区间：bin = floor(y)
        例：y=-1.2 -> -2 对应 [-2,-1)
            y=0.3  -> 0  对应 [0,1)
        只保留图里范围：[-6..8]（即 [-6,-5)...[8,9)）
        超出范围的会被夹到边界 bin。
        """
        b = np.floor(y).astype(int)
        b = np.clip(b, min_bin, max_bin)
        return b

    def split_by_activity_bins_8_2(self, seed=42, min_bin=-6, max_bin=8):
        """
        按活性区间分层，每个区间 8:2 划分 train/val
        返回：train_idx(list), val_idx(list)
        """
        rng = np.random.RandomState(seed)
        y = self.exp.reshape(-1).astype(float)   # (N,)
        bins = self._bin_activity(y, min_bin=min_bin, max_bin=max_bin)

        train_idx, val_idx = [], []
        for b in range(min_bin, max_bin + 1):
            idx = np.where(bins == b)[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)

            n = len(idx)
            n_val = int(round(n * 0.2))

            # 小桶兜底：至少留 1 个做 val（当 n>=2），否则全进 train
            if n >= 2:
                n_val = max(1, n_val)
            else:
                n_val = 0

            val_idx.extend(idx[:n_val].tolist())
            train_idx.extend(idx[n_val:].tolist())

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        print(f"[split] total={len(y)} train={len(train_idx)} val={len(val_idx)}")
        return train_idx, val_idx


    def data_load1(self,dataset):  # open the first biological experimental result
        # predeal part,load the file
        # path = './data/'+dataset
        data = open("/root/autodl-tmp/cxy/CODE/STR/data/ecoli_prediction.csv", 'r')
        next(data)
        seq = []
        exp = []
        for item in data:
            item = item.split()
            seq.append(item[0])
            exp.append(item[1])
        data.close()

        # transform the exp into array format
        expression = np.zeros((len(exp), 1))
        i = 0
        while i < len(exp):
            expression[i] = float(exp[i])
            i = i + 1

        return seq, expression


    def seq_onehot(self,seq):
        # seq, expression = self.data_load(self.dataset)
        # print(seq)
        # print(expression)
        # print(len(seq))
        onehot_seq=[]
        for i in seq:
            one_hot_matrix = self.one_hot_encode(self.string_to_array(i))
            onehot_seq.append(torch.Tensor(one_hot_matrix))


        # max_length = max(i.shape[-1] for i in onehot_seq)
        max_length = 4
        min_length = min(i.shape[-1] for i in onehot_seq)
        # print("max_length:",max_length)
        # print("min_length:",min_length)

        # 填充到最大长度，并使用特定的填充值（例如 0）进行填充
        padded_tensor_list = []
        for i in onehot_seq:
            padding_length = max_length - i.shape[-1]
            padded_tensor = pad(i, (0, padding_length), value=0)
            padded_tensor_list.append(padded_tensor)

        # 使用 pad_sequence 将填充后的张量列表组合成一个批次
        onehot_seq = pad_sequence(padded_tensor_list, batch_first=True)

        # print(onehot_seq)
        input_size = onehot_seq.shape[-1]
        # print("input_size:",input_size)

        return onehot_seq


    def build_model(self):
        self.model = GRUModel(self.input_size, self.hidden_size, self.output_size, dropout_rate=0.2, lambda_l2=0.001)
        if self.use_gpu:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=self.lambda_l2)
        self.criterion = nn.MSELoss()

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.checkpoint_dir + "model_weights_{}.pth".format(epoch))

    def load_model(self):
        '''
            Load model parameters from most recent epoch
        '''
        list_model = glob.glob(self.checkpoint_dir + "model*.pth")
        if len(list_model) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        chk_file = max(list_model, key=os.path.getctime)
        epoch_found = int( (chk_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found!".format(epoch_found))
        self.model.load_state_dict(torch.load(chk_file))
        return epoch_found


    def train(self):

        # Split training/validation and testing set
        expression = self.exp
        onehot_seq = self.seq

        seq = onehot_seq
        train_feature = seq[self.train_idx]
        train_label = expression[self.train_idx]

        # 将数据转换为Tensor类型
        train_feature = torch.Tensor(train_feature).cuda()
        train_label = torch.Tensor(train_label).cuda()


        train_data = TensorDataset(train_feature, train_label)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)


        # 训练模型
        num_epochs = 100 # 自定义训练轮数
        loss_total = 0.0
        num_batch = int(len(seq)/self.batch_size)
        best_loss = float('inf')  # 初始化为正无穷大
        # 梯度裁剪阈值
        clip_value = 1.0
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.save_path + 'predictor.pth', stop_order='max')

        weighted_loss_function = WeightedMSELoss(weight=torch.Tensor([10.0]).cuda())

        with open('/root/autodl-tmp/cxy/CODE/STR/prediction/running/onlyGRUaccuracy.txt', 'w') as f:
            for epoch in range(num_epochs):
                # self.model.train()
                epoch_loss = 0.0
                for batch_idx, (batch_feature, batch_label) in enumerate(train_loader):
                    #enumerate(train_loader)用于在训练循环中同时迭代训练加载器中的批次和批次索引
                    #batch_idx会自动递增，指示当前处理的是第几个批次
                    batch_feature = batch_feature.cuda()
                    batch_label = batch_label.cuda()

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        output = self.model(batch_feature)
                        # loss = self.criterion(output, batch_label)
                        loss = weighted_loss_function(output, batch_label) # 使用加权损失函数

                    loss.backward()

                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                    self.optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                print("Epoch:", epoch, "Loss:", avg_loss)
                if(avg_loss < best_loss):
                    best_loss = avg_loss
                    self.save_model(epoch)
                loss_total += avg_loss


                rho, cor, mse = self.evaluate()
                # f.write(f"Epoch {epoch}: Accuracy {rho}\n")
                f.write(f"rho:{rho},cor:{cor},mse:{mse}\n")
                early_stopping(val_loss=rho, model=self.model)
                if early_stopping.early_stop:
                    print('Early Stopping......')
                    break

        plt.plot(range(1, len(self.val_acc_list) + 1), self.val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.legend()
        plt.show()



        # avg_loss_total = loss_total / num_epochs
        # print("Average Loss:", avg_loss_total)

    def evaluate(self):
        # expression = self.exp
        # onehot_seq = self.seq
        # seq = onehot_seq
        # r = self.r    #设置训练样本的数量
        # test_feature = seq[r:len(seq)]  #从数据中划分测试集特征
        # test_label = expression[r:len(seq)]  #从表达量数据中划分测试集标签
        # test_feature = torch.Tensor(test_feature).cuda()


        seq = self.seq
        exp = self.exp
        test_feature = seq[self.val_idx]
        test_label = exp[self.val_idx]

        test_feature = torch.Tensor(test_feature).cuda()


        # 在测试集上进行预测
        # self.model.eval()
        with torch.no_grad():
            test_output = self.model(test_feature)

        # 将Tensor转换为numpy数组
        test_output = test_output.cpu()
        y_pred = test_output.numpy()
        print("test_lable:",test_label,"y_pred:",y_pred)
        print("y_pred_shape:",len(y_pred))

        # 计算均方误差（Mean Squared Error）
        mse = mean_squared_error(test_label, y_pred)
        print('Mean Squared Error:', mse)
        cor_pearsonr = pearsonr(test_label, y_pred)[0]
        print("cor_pearson:",cor_pearsonr)

        # correlation_matrix = np.corrcoef(test_label, y_pred)
        # corrcoef = correlation_matrix[0, 1]
        # print("co",corrcoef)


        # 计算斯皮尔曼相关系数
        rho, p_value = spearmanr(test_label, y_pred)
        print("斯皮尔曼相关系数:", rho)
        self.val_acc_list.append(rho)

        # 计算额外的评估指标
        mae = mean_absolute_error(test_label, y_pred)
        explained_var = explained_variance_score(test_label, y_pred)

        print('Mean Absolute Error:', mae)
        print('Explained Variance Score:', explained_var)
        return rho,cor_pearsonr,mse


#预测与验证
    def valdata1(self):
        self.model.load_state_dict(torch.load("/root/autodl-tmp/cxy/CODE/STR/results/model/predictor_state_dict.pth"))
        self.model.eval()

        df = pd.read_csv(
            "/root/autodl-tmp/cxy/CODE/STR/cachetest2/inducible_fillseq_tiqu_2026-02-04-22-49-17_results.csv")
        # 按你的文件结构选列：通常 fakeB 是生成序列
        inputseq = df["fakeB"].astype(str).tolist()

        valseq_onehot = self.seq_onehot(inputseq)
        valseq = torch.tensor(valseq_onehot, dtype=torch.float32).cuda()

        with torch.no_grad():
            val_output = self.model(valseq)

        val_pred = val_output.cpu().numpy()

        out_df = pd.DataFrame({"seq": inputseq, "pred": val_pred.reshape(-1)})
        out_df.to_csv("/root/autodl-tmp/cxy/CODE/STR/results/val_predictions.csv", index=False)

        return val_pred

    def valdata(self,seq):
        self.model.load_state_dict(torch.load("/root/autodl-tmp/cxy/CODE/STR/results/model/predictor_state_dict.pth"))
        # seq = "GGGTCGATGATTTTGGTTAAACAGCTTGTTGTATTTGCTTTTTAGGTTTATGTCAGCTCTGCATCTCGATTTCCGTCCTGGTTGGGCGCTAGTTTGTCAACGCTAATTGGTGCTACATATTATGGTTTTGAACATTTTTCATTATTTATAAAGGGGGGATTCCTA"
        valseq_onehot = self.seq_onehot([seq])
        valseq = torch.Tensor(valseq_onehot).cuda()

        with torch.no_grad():
            val_output = self.model(valseq)

        # 将Tensor转换为numpy数组
        val_output = val_output.cpu()
        val_pred = val_output.numpy()
        # print("val_pred:",val_pred)
        # print("y_pred_shape:",len(val_pred))
        return val_pred





if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间

    predict = PREDICT()
    #predict.train()
    # predict.evaluate()

    predict.valdata1()
    # seq = "GGGTCGATGATTTTGGTTAAACAGCTTGTTGTATTTGCTTTTTAGGTTTATGTCAGCTCTGCATCTCGATTTCCGTCCTGGTTGGGCGCTAGTTTGTCAACGCTAATTGGTGCTACATATTATGGTTTTGAACATTTTTCATTATTTATAAAGGGGGGATTCCTA"
    # predict.valdata(seq)

    # avg_rho = predict.cross_validation()
    # print("Average Spearman correlation:", avg_rho)

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)