# python /root/autodl-tmp/cxy/CODE/STR/Generator/loss.py

#这里直接实例化一个新的判别器？

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


class SequenceDatasetNatural(Dataset):
    def __init__(self, file_path):
        self.sequences = list(pd.read_csv(file_path)['realB'])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence_tensor = torch.tensor(one_hot(sequence), dtype=torch.float32)
        return sequence_tensor

class SequenceDatasetSTR(Dataset):
    def __init__(self, file_path):
        self.sequences = list(pd.read_csv(file_path, header=None)[0])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence_tensor = torch.tensor(one_hot(sequence), dtype=torch.float32)
        return sequence_tensor


def one_hot(sequence):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoding = np.zeros((4, len(sequence)))
    for i, base in enumerate(sequence):
        encoding[charmap[base], i] = 1
    return encoding


def backbone_one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded


class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=13, padding=6, bias=True):
        super(ResBlock, self).__init__()
        model = [
            nn.ReLU(inplace=False),
            nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=False),
            nn.Conv1d(output_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print("x.shape:",x.shape)
        x1 = self.model(x)
        # print("x1.shape:",x1.shape)
        x2 = x + 0.3 * x1
        # print("x2.shape:",x2.shape)
        return x2

class Discriminator(nn.Module):
    def __init__(self, input_channels=4, output_channels=1, ndf=163):
        super(Discriminator, self).__init__()

        self.conv1d = nn.Conv1d(input_channels * 2, ndf, kernel_size=3, stride=1)
        # ResBlock层
        model = [ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf), ]
        self.model = nn.Sequential(*model)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ndf*ndf, output_channels)

    def forward(self, sequence1, sequence2):
        x = torch.cat((sequence1, sequence2), dim=1)  # Concatenate along channel axis
        # print("Shape before linear layer1:", x.shape)  # 打印形状

        x = self.conv1d(x)
        # print("Shape before linear layer1:", x.shape)  # 打印形状

        x = self.model(x)
        # print("Shape before linear layer1:", x.shape)  # 打印形状

        x = self.relu(x)
        # print("Shape before linear layer2:", x.shape)  # 打印形状

        x = self.flatten(x)
        # print("Shape before linear layer3:", x.shape)  # 打印形状
        output = self.linear(x)
        # print("Shape after linear layer4:", output.shape)  # 打印形状
        return output

# class CNNSequenceDiscriminator(nn.Module):
#     def __init__(self, input_channels=4, output_channels=1, ndf=163):
#         super(CNNSequenceDiscriminator, self).__init__()
#         self.conv1d = nn.Conv1d(input_channels * 2, output_channels, kernel_size=3, stride=1)
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(ndf, output_channels)
#
#     def forward(self, sequence1, sequence2):
#         x = torch.cat((sequence1, sequence2), dim=1)  # Concatenate along channel axis
#         x = self.conv1d(x)
#         print("Shape before linear layer1:", x.shape)  # 打印形状
#         x = self.relu(x)
#         print("Shape before linear layer2:", x.shape)  # 打印形状
#         x = self.flatten(x)
#         print("Shape before linear layer3:", x.shape)  # 打印形状
#         output = self.linear(x)
#         print("Shape after linear layer4:", output.shape)  # 打印形状
#         return output


def calculate_sequence_loss(file1_path, file2_path, batch_size=1):
    # Load datasets
    dataset1 = SequenceDatasetSTR(file1_path)
    dataset2 = SequenceDatasetNatural(file2_path)
    # print("dataset1:",dataset1)
    # print("dataset2:",dataset2)

    # Create data loaders
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    # Instantiate the discriminator
    discriminator = Discriminator()

    # Specify device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    discriminator.to(device)

    # Loss criterion (Mean Squared Error in this case)
    criterion = nn.MSELoss()

    # List to store the calculated losses
    losses = []

    with torch.no_grad():
        for data1, data2 in zip(dataloader1, dataloader2):
            # Convert sequences to tensors
            data1, data2 = torch.tensor(data1, dtype=torch.float32), torch.tensor(data2, dtype=torch.float32)

            # Move data to the device
            data1, data2 = data1.to(device), data2.to(device)

            # Forward pass through the discriminator
            output = discriminator(data1, data2)

            # Calculate the loss
            loss = criterion(output, torch.zeros_like(output))  # Assuming zero as the target for simplicity

            # Append the sequence and loss to the list
            losses.append(loss.item())

    return losses

def main():
    str1_path = "/root/autodl-tmp/cxy/CODE/STR/cachetest1/fillseq_20260203095533.csv"        #经STR嵌入后的第一阶段生成序列（5920条）
    nat2_path = "/root/autodl-tmp/cxy/CODE/STR/data/ecoli_generation.csv"                    #天然序列（5920条）
    losses = calculate_sequence_loss(str1_path, nat2_path, batch_size=1)
    with open(str1_path,'r') as f, open(str1_path.replace('.csv','_with_loss.csv'),'w') as outputfile:
        lines = f.readlines()
        #lines = f.readlines()[1:]

        data = [line.strip().split(",")[0] for line in lines]
        m = [line.strip().split(",")[1] for line in lines]
        pred = [line.strip().split(",")[2] for line in lines]
        realA1 = [line.strip().split(",")[3] for line in lines]
        realB1 = [line.strip().split(",")[4] for line in lines]

        for seq, hm, loss, exp , realA, realB in zip(data, m, losses, pred, realA1, realB1):
            outputfile.write(seq + ',' + hm + ',' + str(loss) + ',' + exp + ',' + realA + ',' + realB + '\n')


    # 读取文件并按表达值大小排序
    with open(str1_path.replace('.csv','_with_loss.csv'), "r") as f:
        lines = f.readlines()

        # 使用lambda函数和sorted函数根据表达值列排序
        sorted_lines = sorted(lines, key=lambda x: float(x.split(',')[2]), reverse=False)

    # 写入排序后的结果到新文件
    with open(str1_path.replace('.csv','_with_loss_sort.csv'), 'w') as f:
        # f.write(header)
        f.writelines(sorted_lines)


    with open(str1_path.replace('.csv','_with_loss_sort.csv'), "r") as input_file, open(str1_path.replace('.csv','_with_loss_sort1.csv'),"w") as output_file:
        for line in input_file:
            first_colum = line.strip().split(',')[0]

            print(first_colum)
            output_file.write(first_colum + '\n')


     # 读取原始数据文件
    with open(str1_path.replace('.csv','_with_loss_sort1.csv'), "r") as input_file:
        data_content = input_file.read()

    # 分割序列
    sequences = data_content.split("\n")

    # 生成FASTA格式内容
    fasta_content = ""
    for i, sequence in enumerate(sequences):
        if sequence:
            fasta_content += f">{i}\n{sequence}\n"

    # 将FASTA内容写入文件
    with open(str1_path.replace('.csv','_with_loss_sort.fasta'), "w") as fasta_file:
        fasta_file.write(fasta_content)

    print("FASTA文件已生成")
    sorted_file_path = str1_path.replace('.csv', '_with_loss_sort.csv')
    tiqu(sorted_file_path)


def tiqu(input_path):
    with open(input_path,'r') as f, open(input_path.replace('_sort.csv','_tiqu.csv'),'w') as outputfile:
        lines = f.readlines()
        num_lines = len(lines)
        num_to_extract = int(num_lines * 1)

        condition = [line.strip().split(",")[1] for line in lines[:num_to_extract]]
        realB1 = [line.strip().split(",")[0] for line in lines[:num_to_extract]]

        outputfile.write('fake,realB' + '\n')

        for fake, realB in zip(condition, realB1):
            outputfile.write(fake + ',' + realB + '\n')

if __name__ == '__main__':
    main()

