#生成大肠杆菌的随机序列
import random

def generate_random_sequence(length):
    bases = ['A', 'T', 'C', 'G']
    probabilities = [0.25, 0.25, 0.25, 0.25]  # 设置每个碱基的出现概率
    sequence = ''.join(random.choices(bases, weights=probabilities, k=length))
    return sequence

# 生成长度为165的随机碱基序列
with open("/home/cxm/train/seq-exp/rannatger/raneditdata.txt", "w") as outfile:
    for _ in range(100):
        random_sequence = generate_random_sequence(165)
        outfile.write(random_sequence + '\n')
# #
# #  # 读取原始数据文件
# # with open("/home/cxm/train/seq-exp/rannatger/Random200.txt", "r") as input_file:
# #     data_content = input_file.read()
# #
# # # 分割序列
# # sequences = data_content.split("\n")
# #
# # # 生成FASTA格式内容
# # fasta_content = ""
# # for i, sequence in enumerate(sequences):
# #     if sequence:
# #         fasta_content += f">{i}\n{sequence}\n"
# #
# # # 将FASTA内容写入文件
# # with open("/home/cxm/train/seq-exp/rannatger/ran500.fasta", "w") as fasta_file:
# #     fasta_file.write(fasta_content)
# #
# # print("FASTA文件已生成")
#
#提取<五或者>五的数据
# def filter_data(input_file, output_file):
#     with open(input_file, 'r') as file_in:
#         with open(output_file, 'w') as file_out:
#             next(file_in)  # 跳过第一行
#             for line in file_in:
#                 parts = line.strip().split(',')
#                 realA = str(parts[0])
#                 realB = str(parts[1])
#                 expr = float(parts[2])
#                 if expr < 5:
#                     file_out.write(f"{realA},{realB},{expr}\n")
#
# input_file_path = "/home/cxm/train/seq-exp/data/ecoli_mpra_3_laco.csv" # 修改为你的输入文件路径
# output_file_path = "/home/cxm/train/seq-exp/data/expxy5.csv"  # 修改为你的输出文件路径
# filter_data(input_file_path, output_file_path)
