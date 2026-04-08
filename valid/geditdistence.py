import numpy as np
import matplotlib.pyplot as plt
import random

def gene_edit_distance(seq1, seq2):
    m = len(seq1)
    n = len(seq2)

    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

def read_sequences_from_file(file_path, sample_size=100):
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file.readlines()]

    if len(sequences) > sample_size:
        sequences = random.sample(sequences, sample_size)

    print(f"Read {len(sequences)} sequences from {file_path}")
    return sequences

def calculate_edit_distances(sequences):
    num_sequences = len(sequences)
    distances = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            distance = gene_edit_distance(sequences[i], sequences[j])
            distances[i][j] = distance
            distances[j][i] = distance

    print(f"Calculated distances for {num_sequences} sequences")
    return distances.flatten()

def plot_histograms(natural_distances,gen_distances, random_distances):
    plt.hist(natural_distances, bins=20, density=True, alpha=1, label='Natural', color='#AF97B6')
    plt.hist(gen_distances, bins=20, density=True, alpha=0.8, label='STR-designed', color='#C6D180')
    plt.hist(random_distances, bins=20, density=True, alpha=0.8, label='Random', color='#e3716e')


    plt.xlabel('Edit Distance')
    plt.ylabel('Frequency Distribution')
    plt.legend()
    plt.title('Edit Distance Distribution')
    plt.savefig('/home/cxm/train/seq-exp/results/editdata/edit_histogram.png')
    plt.show()
    plt.close()

import seaborn as sns

def plot_density(natural_distances, gen_distances, random_distances):
    # 使用 seaborn 绘制核密度曲线
    sns.kdeplot(natural_distances, label='Natural', color='#AF97B6', fill=False, linewidth = 5)
    sns.kdeplot(gen_distances, label='STR-designed', color='#C6D180', fill=False, linewidth = 5)
    sns.kdeplot(random_distances, label='Random', color='#e3716e', fill=False, linewidth = 5)

    plt.xlabel('Distance Range',fontsize=18)
    plt.ylabel('Frequency',fontsize=18)
    plt.legend()
    # plt.title('Edit Distance Distribution',fontsize=16)
    plt.xlim(60,130)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=18)  # 增加坐标轴数值字体大小
    # # 设置 y 轴刻度
    # yticks = np.arange(0, 0., 0.03)  # 根据需要调整范围和间隔
    # plt.yticks(yticks)
    plt.savefig('/home/cxm/train/seq-exp/results/editdata/edit_density.png')
    plt.show()
    plt.close()

def plot_curves(natural_distances, random_distances, gen_sequences):
    plt.plot(np.sort(natural_distances), np.linspace(0, 1, len(natural_distances), endpoint=False),
             label='Natural', color='#AF97B6')
    plt.plot(np.sort(gen_sequences), np.linspace(0, 1, len(gen_sequences), endpoint=False),
             label='STR-designed', color='#C6D180')
    plt.plot(np.sort(random_distances), np.linspace(0, 1, len(random_distances), endpoint=False),
             label='Random', color='#e3716e')


    plt.xlabel('Edit Distance')
    plt.ylabel('Frequency Distribution')
    plt.legend()
    plt.title('Edit Distance Distribution')
    # 设置 x 轴的区间
    plt.xlim(60, 130)
    plt.savefig('/home/cxm/train/seq-exp/results/editdata/edit_curve.png')

    plt.close()

# Example file paths (replace these with your actual file paths)
natural_file_path = "/home/cxm/train/seq-exp/rannatger/nateditdata.txt"
random_file_path = "/home/cxm/train/seq-exp/rannatger/raneditdata.txt"
gen_sequences = "/home/cxm/train/seq-exp/Generator/cacheall/tiquGen.txt"


natural_sequences = read_sequences_from_file(natural_file_path)
gen_sequences = read_sequences_from_file(gen_sequences)
random_sequences = read_sequences_from_file(random_file_path)

# Calculate edit distances
natural_distances = calculate_edit_distances(natural_sequences)
gen_distances = calculate_edit_distances(gen_sequences)
random_distances = calculate_edit_distances(random_sequences)

# Plot histograms
plot_histograms(natural_distances, gen_distances, random_distances)

# Plot histograms
plot_density(natural_distances, gen_distances, random_distances)

# Plot curves
plot_curves(natural_distances, gen_distances, random_distances)
