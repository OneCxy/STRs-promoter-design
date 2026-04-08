import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

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

def calculate_edit_distances(seq1_list, seq2_list):
    num_seq1 = len(seq1_list)
    distances = np.zeros((num_seq1, len(seq2_list)))

    for i in range(num_seq1):
        for j in range(len(seq2_list)):
            distance = gene_edit_distance(seq1_list[i], seq2_list[j])
            distances[i][j] = distance

    print(f"Calculated distances for {num_seq1} sequences against {len(seq2_list)} sequences")
    return distances

def plot_histograms(distances_list, labels):
    plt.figure(figsize=(12, 8))
    for distances, label in zip(distances_list, labels):
        plt.hist(distances.flatten(), bins=20, density=True, alpha=0.5, label=f'Histogram: {label}')

    plt.xlabel('Edit Distance')
    plt.ylabel('Frequency Distribution')
    plt.title('Edit Distance Distribution - Histograms')
    plt.legend()
    # plt.grid()
    plt.savefig('/home/cxm/train/seq-exp/results/editdata/edit_distance_histogram1v1.png')
    plt.close()

def plot_density(distances_list, labels):
    plt.figure(figsize=(12, 8))
    for distances, label in zip(distances_list, labels):
        sns.kdeplot(distances.flatten(), label=f'Density: {label}', fill=False, linewidth=6)

    plt.xlabel('Edit Distance', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    # plt.title('Edit Distance', fontsize=16)
    plt.xlim(60, 130)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=24)  # 增加坐标轴数值字体大小
    # 设置 y 轴刻度
    yticks = np.arange(0, 0.12, 0.03)  # 根据需要调整范围和间隔
    plt.yticks(yticks)
    # plt.grid()
    plt.savefig('/home/cxm/train/seq-exp/results/editdata/edit_distance_density1v1.png')
    plt.close()

# Example file paths (replace these with your actual file paths)
natural_file_path = "/home/cxm/train/seq-exp/rannatger/nateditdata.txt"
random_file_path = "/home/cxm/train/seq-exp/rannatger/raneditdata.txt"
gen_file_path = "/home/cxm/train/seq-exp/Generator/cacheall/tiquGen.txt"

# Read sequences
natural_sequences = read_sequences_from_file(natural_file_path)
random_sequences = read_sequences_from_file(random_file_path)
gen_sequences = read_sequences_from_file(gen_file_path)

# Calculate edit distances
random_gen_distances = calculate_edit_distances(random_sequences, gen_sequences)
random_nat_distances = calculate_edit_distances(random_sequences, natural_sequences)
gen_nat_distances = calculate_edit_distances(gen_sequences, natural_sequences)

# Prepare data for plotting
distances_list = [random_gen_distances, random_nat_distances, gen_nat_distances]
labels = ['Random vs Generated', 'Random vs Natural', 'Generated vs Natural']

# Plot histograms
plot_histograms(distances_list, labels)

# Plot densities
plot_density(distances_list, labels)

