import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D, self).__init__()

        self.in_channels = in_channels #通道数
        self.inter_channels = inter_channels #中间层数

        if self.inter_channels is None: # 若没提供，默认是通道数的一半
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            # nn.BatchNorm1d(self.inter_channels),
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x) # 计算点积，生成注意力矩阵f，该矩阵描述了输入特征在时间维度上的依赖关系
        f_div_C = F.softmax(f, dim=-1) # 将f通过softmax函数，使其转化为注意力权重，保留输入特征的重要信息

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # 输出y，包含输入特征的长距离依赖关系
        W_y = self.W(y) # 通过卷积层W将y应社会输入通道数的维度，生成最终的输出特征
        z = W_y + x # 将映射后的特征和输入特征相加，得到非局部块的最终输出z

        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 线性变换
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # 分割成多个头
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())  # (batch_size, num_heads, seq_length, seq_length)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 注意力权重
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_length, head_dim)

        # 合并多个头
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_length, d_model)

        # 线性变换
        output = self.output_linear(attention_output)  # (batch_size, seq_length, d_model)

        return output



def main():
    # 准备输入数据
    query = torch.randn(2, 10, 64)  # 2个样本，每个样本10个查询向量，每个查询向量维度为64
    key = torch.randn(2, 12, 64)  # 2个样本，每个样本12个键向量，每个键向量维度为64
    value = torch.randn(2, 12, 64)  # 2个样本，每个样本12个值向量，每个值向量维度为64

    # 创建多头注意力模型实例
    d_model = 64  # 输入向量的维度
    num_heads = 4  # 注意力头数
    attention = MultiHeadAttention(d_model, num_heads)

    # 执行前向传播
    output, attention_weights = attention(query, key, value)

    # 打印输出结果
    print("输出形状:", output.shape)
    print("注意力权重形状:", attention_weights.shape)

if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     import torch
#
#     img = torch.zeros(2, 3, 20)
#     net = NonLocalBlock1D(3)
#     out = net(img)
#     print(out.size())
