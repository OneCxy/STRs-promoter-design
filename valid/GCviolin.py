import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_gc_content(sequences):
    gc_contents = []
    for seq in sequences:
        gc_count = seq.count('G') + seq.count('C')
        total_count = len(seq)
        gc_content = gc_count / total_count if total_count > 0 else 0
        gc_contents.append(gc_content)
    return gc_contents

def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# 文件路径
natural_file = "/home/cxm/train/seq-exp/rannatger/nateditdata.txt"
random_file = "/home/cxm/train/seq-exp/rannatger/raneditdata.txt"
generated_file = "/home/cxm/train/seq-exp/rannatger/geneditdata.txt"

# 读取序列
generated_sequences = read_sequences_from_file(generated_file)
natural_sequences = read_sequences_from_file(natural_file)
random_sequences = read_sequences_from_file(random_file)

# 计算 GC 含量
gc_generated = calculate_gc_content(generated_sequences)
gc_natural = calculate_gc_content(natural_sequences)
gc_random = calculate_gc_content(random_sequences)

# 准备数据用于小提琴图
data = {
    'Generated': gc_generated,
    'Natural': gc_natural,
    'Random': gc_random
}

# 绘制小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(data=list(data.values()),showmedians=True, inner=None, bw='scott', scale='width', alpha=0.3, showextrema=False)

# 计算中位数和四分位数
medians = [np.median(gc) for gc in data.values()]
q1 = [np.percentile(gc, 25) for gc in data.values()]
q3 = [np.percentile(gc, 75) for gc in data.values()]

# 绘制中位数和四分位线
ind = np.arange(len(data))

plt.vlines(ind, q1, q3, color='black', lw=5)
plt.scatter(ind, medians, color='white', marker='o', s=100,zorder=3)

sns.boxplot(data=list(data.values()),
            color='black',
            boxprops=dict(edgecolor='black', linewidth=1.5),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black'),
            width=0.1,  # 减小箱线图的宽度
            fliersize=2)
# 设置 x 轴标签和标题
plt.xticks(ticks=ind, labels=data.keys(), fontsize=20)
plt.ylabel('GC Content', fontsize=20)
# plt.title('GC Content Distribution in Different Sequence Types')
# plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=24)  # 增加坐标轴数值字体大小
# 增加坐标轴粗细
plt.tick_params(axis='both', which='major', width=4)  # 主刻度线宽
plt.tick_params(axis='both', which='minor', width=4)  # 次刻度线宽
# 保存和显示图形
plt.savefig('/home/cxm/train/seq-exp/results/GCdata/gc_content_violin_box_plot.png')
plt.show()
plt.close()