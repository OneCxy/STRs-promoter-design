import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = """

"""

data_lines = data.split('\n')

genomic_data = []
natural_data = []
random_data = []
current_category = None

for line in data_lines:
    if line.startswith('>Gen'):
        current_category = 'genomic_data'
    elif line.startswith('>Nat'):
        current_category = 'natural_data'
    elif line.startswith('>Ran'):
        current_category = 'random_data'
    else:
        if current_category == 'genomic_data':
            genomic_data.extend([float(val) for val in line.split(',')[1:] if val != 'NA'])
        elif current_category == 'natural_data':
            natural_data.extend([float(val) for val in line.split(',')[1:] if val != 'NA'])
        elif current_category == 'random_data':
            random_data.extend([float(val) for val in line.split(',')[1:] if val != 'NA'])

print("Genomic Data:", genomic_data)
print("Natural Data:", natural_data)
print("Random Data:", random_data)


def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

genomic_data_normalized = normalize_data(genomic_data)
natural_data_normalized = normalize_data(natural_data)
random_data_normalized = normalize_data(random_data)

sns.kdeplot(genomic_data_normalized, color='blue', label='Generated', linewidth=6, linestyle='-')
sns.kdeplot(natural_data_normalized, color='green', label='Natural', linewidth=6, linestyle='-')
sns.kdeplot(random_data_normalized, color='orange', label='Random', linewidth=6, linestyle='-')


# plt.xlabel('DNA Shape Parameter',fontsize=18)
plt.ylabel('Frequency',fontsize=24)
plt.title('ROLL,TATAAA',fontsize=22)
# plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=20)  # 增加坐标轴数值字体大小
plt.show()
