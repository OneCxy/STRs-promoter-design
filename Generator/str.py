# python /root/autodl-tmp/cxy/CODE/STR/Generator/str.py

import re
import random
import string
import time

import pandas as pd

import numpy as np
import sys
import os

sys.path.append('/root/autodl-tmp/cxy/CODE/STR')
from prediction.GRU import PREDICT




def extract_motifs(seq1,seq2):
    # 使用正则表达式提取所有非"M"的子序列（基序）
    motifs = re.findall(r'[^M]+', seq1)

    # 提取每个基序的起始和结束位置
    motif_positions = [(match.start(), match.end()) for match in re.finditer(r'[^M]+', seq1)]

    if len(motif_positions) == 0:
        side_sequence_positions = [(0, len(seq2))]
        side_sequence = [seq2]
        print("seq1无基序，全侧翼：", side_sequence_positions)
        return motifs, motif_positions, side_sequence, side_sequence_positions

    side_sequence = []
    side_sequence_positions = []

    if(motif_positions[0][0] != 0):
        side_sequence_positions.append((0, motif_positions[0][0]))

    # 添加每两个基序之间的侧翼序列位置
    for i in range(len(motif_positions) - 1):
        start = motif_positions[i][1]
        end = motif_positions[i + 1][0]

        side_sequence_positions.append((start, end))

    # 添加最后一个基序之后的侧翼序列位置
    side_sequence_positions.append((motif_positions[-1][1], len(seq2)))

    print("侧翼序列位置：",side_sequence_positions)

    # 打印seq2的具体侧翼序列
    for start, end in side_sequence_positions:
        side_sequence.append(seq2[start:end])

    print("侧翼序列：",side_sequence)

    return motifs, motif_positions, side_sequence, side_sequence_positions


def generate_random_str_sequences(start, end):
    # 定义碱基
    bases = ['A', 'T', 'C', 'G']
    while True:
        # 随机选择碱基单元的大小和重复次数
        random.seed(time.time())
        unit_size = random.randint(2, 3)
        repeat_count = random.randint(2, 3)


        # 计算侧翼序列的最大长度
        max_sequence_length = end - start

        if max_sequence_length >= (unit_size * repeat_count):
            print("unit_size:", unit_size)
            print("repeat_count:", repeat_count)

            # 随机选择碱基单元
            selected_bases = random.sample(bases, unit_size)

            # 生成短串联重复序列
            sequence = ''.join(selected_bases) * repeat_count

            return sequence  # 满足条件，返回生成的序列


def fill(seq1, seq2):
    # 提取基序和基序位置
    motifs, motif_positions, side_sequence, side_sequence_positions = extract_motifs(seq1, seq2)

    # 打印结果
    print("基序:", motifs)
    print("基序位置:", motif_positions)

    random_side_sequence = []
    random_m_str_sequence = []  # 修正这里

    for i, (start, end) in enumerate(side_sequence_positions):
        if (end - start) <= 4:
            random_str_sequence = side_sequence[i]
        else:
            random_str_sequence = generate_random_str_sequences(start, end)

        print("随机 DNA 重复序列：", random_str_sequence)

        # 如果随机生成的 DNA 重复序列长度比侧翼序列短，用侧翼序列填充剩余部分
        if len(random_str_sequence) < (end - start):
            remaining_sequence = seq2[start + len(random_str_sequence): end]
            random_sequence = random_str_sequence + remaining_sequence
            random_side_sequence.append(random_sequence)
        else:
            random_side_sequence.append(random_str_sequence)  # 修正这里


        if len(random_str_sequence) < (end - start):  # 修正这里
            remaining_m_sequence = "M" * (end - start - len(random_str_sequence))
            random_m_str_sequence.append(random_str_sequence + remaining_m_sequence)  # 修正这里
        else:
            random_m_str_sequence.append(random_str_sequence)  # 修正这里

    # 打印seq2的具体侧翼序列
    for i, (start, end) in enumerate(side_sequence_positions):
        seq2 = seq2[:start] + random_side_sequence[i] + seq2[end:]

    # 对于seq1，如果随机生成的 DNA 重复序列长度比侧翼序列短，用"M"填充剩余部分
    for i, (start, end) in enumerate(side_sequence_positions):
        seq1 = seq1[:start] + random_m_str_sequence[i] + seq1[end:]

    print("修改后的seq2：", seq2)
    print("修改后的seq1：", seq1)

    return seq2, seq1


def dataload():
    files = pd.read_csv("/root/autodl-tmp/cxy/CODE/STR/cachetest/inducible_ecoli_2026-02-03-00-11-37_results.csv")
    realA = list(files['realA'])
    realB = list(files['realB'])
    fakeB = list(files['fakeB'])
    pred = list(files['predict'])
    print("realA",len(realA))
    print("fakeB",len(fakeB))
    print("pred",len(pred))
    predict = PREDICT()
    seq = dict(zip(realA,fakeB))
    print(seq)
    print("df.shape", files.shape)
    print("realB", len(realB))

    # 添加时间戳
    timestamp = time.strftime("%Y%m%d%H%M%S")
    count_update = 0

    with open(f'/root/autodl-tmp/cxy/CODE/STR/cachetest1/fillseq_{timestamp}.csv', 'w') as f:
        for seq1,seq2,pred_val,seq3 in zip(realA, fakeB, pred, realB):
            # 使用正则表达式提取字符串中的数字
            pred_val_float = float(re.search(r'[-+]?\d*\.\d+|\d+', pred_val).group())
            best_fillseq2 = seq2
            best_pred_str = predict.valdata(seq2)[0]
            best_fillseqM = seq1

            for _ in range(5):  # 重复五次
                fillseq2,fillseqM = fill(seq1, seq2)
                new_pred = predict.valdata(fillseq2)[0]
                print("修改后的预测值:", new_pred)
                if new_pred >= best_pred_str:
                    best_fillseq2 = fillseq2
                    best_fillseqM = fillseqM
                    best_pred_str = new_pred

            if best_pred_str >= pred_val_float:
                f.write(str(best_fillseq2)+ ',' + str(best_fillseqM) + ',' + str(best_pred_str) + ',' + str(seq1) + ',' + str(seq3) + ', updata' + '\n')
                count_update += 1
            else:
                f.write(str(seq2) + ',' + str(best_fillseqM) + ',' + str(pred_val_float) + ',' + str(seq1) + ',' + str(seq3) + '\n')
    print("update个数：",count_update)
    return best_fillseq2



def dataload1():
    files = pd.read_csv("/root/autodl-tmp/cxy/CODE/STR/data/ecoli_generation.csv")
    realA = list(files['realA'])
    realB = list(files['realB'])
    expr = list(files['expr'])
    print("realA",len(realA))
    print("expr",len(expr))
    predict = PREDICT()
    seq = dict(zip(realA,realB))
    print(seq)

    # 添加时间戳
    timestamp = time.strftime("%Y%m%d%H%M%S")
    count_update = 0

    with open(f'/root/autodl-tmp/cxy/CODE/STR/cachenat/fillseq_{timestamp}.csv', 'w') as f:
        for seq1,seq2,expr_val in zip(realA, realB, expr):
            # 使用正则表达式提取字符串中的数字
            # pred_val_float = float(re.search(r'[-+]?\d*\.\d+|\d+', expr_val).group())
            pred_val_float = float(expr_val)
            best_fillseq2 = seq2
            # best_pred_str = predict.valdata(seq2)[0]
            best_pred_str = 0

            for _ in range(5):  # 重复五次
                fillseq2,fillseqM = fill(seq1, seq2)
                new_pred = predict.valdata(fillseq2)[0]
                print("修改后的预测值:", new_pred)

                best_fillseqM = seq1

                if new_pred >= best_pred_str:
                    best_fillseq2 = fillseq2
                    best_fillseqM = fillseqM
                    best_pred_str = new_pred

            if best_pred_str >= expr_val:
                f.write(str(best_fillseq2)+ ',' + str(best_fillseqM) + ',' + str(best_pred_str) + ',' + str(seq1) + ',' + str(seq2) + ','+ 'update' + '\n')
                count_update += 1
            else:
                f.write(str(seq2) + ',' + str(best_fillseqM) + ',' + str(pred_val_float) + ',' + str(seq1) + ',' + str(seq2) + '\n')
    print("update个数：",count_update)
    return best_fillseq2


if __name__ == '__main__':
    dataload()
    # dataload1()

