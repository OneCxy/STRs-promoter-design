import matplotlib.pyplot as plt
import seaborn as sns
from pandas import np


def calculate_edit_distance(seq1, seq2):
    """
    计算两个序列之间的编辑距离。
    """
    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 初始化 DP 矩阵
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 动态规划填表
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2]

def read_file(file_path):
    """
    读取文件中的 DNA 序列。
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def calculate_distances(file1, file2):
    """
    计算两个文件中每对序列的编辑距离。
    """
    seqs1 = read_file(file1)
    seqs2 = read_file(file2)

    if len(seqs1) != len(seqs2):
        raise ValueError("两个文件中的序列数量不一致！")

    distances = []
    for s1, s2 in zip(seqs1, seqs2):
        distances.append(calculate_edit_distance(s1, s2))

    return distances

def plot_kde_distances(results):
    """
    使用核密度曲线图表示编辑距离。
    """
    plt.figure(figsize=(10, 6))
    colors = {
        "Generated" : "blue",
        "Random" : "green",
        "Natural" : "red"
    }

    for label, distances in results.items():
        sns.kdeplot(distances, label=label, color=colors[label], fill=False, alpha=0.8,linewidth=6)

    plt.xlim(60,120)
    plt.xlabel("Edit Distance", fontsize=24)  # 设置x轴字体大小
    plt.ylabel("Density", fontsize=24)  # 设置y轴字体大小
    # plt.title("KDE of Edit Distances Between DNA Sequences", fontsize=16)  # 设置标题字体大小
    plt.legend(fontsize=16)  # 设置图例字体大小
    plt.tick_params(axis='both', labelsize=24)  # 增加坐标轴数值字体大小
    # 设置 y 轴刻度
    yticks = np.arange(0, 0.12, 0.03)  # 根据需要调整范围和间隔
    plt.yticks(yticks)
    # plt.grid()
    plt.savefig('/home/cxm/train/seq-exp/editdata/edit_distance_density.png')
    plt.show()

if __name__ == "__main__":
    # 定义文件路径
    file_pairs = {
        "Generated": ("/home/cxm/train/seq-exp/editdata/geneditdata.txt", "/home/cxm/train/seq-exp/editdata/ranAndnat.txt"),
        "Random": ("/home/cxm/train/seq-exp/editdata/raneditdata.txt", "/home/cxm/train/seq-exp/editdata/genAndnat.txt"),
        "Natural": ("/home/cxm/train/seq-exp/editdata/nateditdata.txt", "/home/cxm/train/seq-exp/editdata/ranAndgen.txt"),
    }

    results = {}

    # 计算每组文件的编辑距离
    for label, (file1, file2) in file_pairs.items():
        results[label] = calculate_distances(file1, file2)

    # 绘制核密度曲线图
    plot_kde_distances(results)
