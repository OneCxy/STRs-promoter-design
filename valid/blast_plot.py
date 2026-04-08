import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import collections
import pandas as pd

from scipy.stats import norm
#注释掉的是只能分析两个文件的，没有注释的是分析三个文件
#两个
# def open_BLAST(file_name, locate = -4):
#     f = open(file_name,'r')
#     record = []
#     for item in f:
#         if 'Sequences producing significant alignments:' in item:
#             i = 0
#             while i < 5:
#                 tmp = f.readline()
#                 if i == 2:
#                     record.append(float(tmp.split()[locate]))
#                 i = i + 1
#
#     average_e_value = np.mean(np.array(record))
#     return record,average_e_value
# 三个
def open_BLAST(file_name, locate=-4):
    f = open(file_name, 'r')
    record = []
    for item in f:
        if 'Sequences producing significant alignments:' in item:
            i = 0
            while i < 5:
                tmp = f.readline()
                if i == 2:
                    record.append(float(tmp.split()[locate]))
                i = i + 1
    f.close()
    return record


'''
Using e-value to calculate density in each sites
'''
def density(record):
    max_num = max(record)
    record = np.array(record)
    density = gaussian_kde(record)  #Using gaussian_kernel to give the density plot
    density.covariance_factor = lambda : 0.14
    density._compute_covariance()
    return max_num, density

def log_density(record):
    record = np.array(record)
    record = np.log10(record)
    max_num = max(record)
    min_num = min(record)
    density = gaussian_kde(record)  #Using gaussian_kernel to give the density plot
    density.covariance_factor = lambda : 0.15
    density._compute_covariance()
    return min_num, max_num, density

def set_default():
    fig, ax=plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_facecolor("white")

    plt.tick_params(labelsize=12,width = 2, length = 5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.grid(False)
    return fig,ax,plt,labels

#两个
# def plot_density(min_num, max_num, color, **density):
#     fig,ax,plt,labels = set_default()
#     xs = np.linspace(min_num, max_num + 0.1 * max_num, 200)
#     for i,item in enumerate(density):
#         density_now = density[item]
#         plt.plot(xs,density_now(xs),label = item.split('_')[0] + ' ' + item.split('_')[1], color = color[i])
#     return plt
# #两个
# def plot_bar(min_num, max_num, color, **blast_val):
#     fig,ax,plt,labels = set_default()
#     bins = list(range(-20,10))
#     for i,item in enumerate(blast_val):
#         blast_log = np.log10(blast_val[item])
#         n, x, _ = plt.hist(blast_log, alpha=0.8 ,color = color[i], density = True
#                     ,label = item.split('_')[0] + ' ' + item.split('_')[1], bins = bins)
#         x_axis = np.linspace(-20, 10, 100)
#         mean = np.mean(blast_log)
#         sigma = np.std(blast_log)
#         plt.plot(x_axis, norm.pdf(x_axis, mean, sigma), color = color[i])
#     plt.legend(loc="best",prop = {'size':14})
#     return plt
#三个
# def plot_density(min_num, max_num, color, **density):
#     fig, ax, plt, labels = set_default()
#     xs = np.linspace(min_num, max_num + 0.1 * max_num, 200)
#     for i, item in enumerate(density):
#         density_now = density[item]
#         plt.plot(xs, density_now(xs), label=item.replace('_', ' '), color=color[i])
#     plt.legend(loc="best", prop={'size': 14})
#     return plt

def plot_bar(min_num, max_num, color, **blast_val):
    fig, ax, plt, labels = set_default()
    bins = list(range(-40, 10))

    for i, item in enumerate(blast_val):
        blast_log = np.log10(blast_val[item])

        n, x, _ = plt.hist(blast_log, alpha=0.8, color=color[i], density=True,
                           label=item.replace('_', ' '), bins=bins, width=0.8)
        x_axis = np.linspace(-40, 10, 100)
        mean = np.mean(blast_log)
        sigma = np.std(blast_log)
        # plt.plot(x_axis, norm.pdf(x_axis, mean, sigma), color=color[i])
    plt.legend(loc="best", prop={'size': 14})
    return plt

def find_index(record,*dataframe):
    for tmp_dataframe in dataframe:
        for row in tmp_dataframe.index:
            if row not in record:
                record.append(row)

def add_index(all_index,*dataframe):
    for tmp_frame in dataframe:
        for tmp_index in all_index:
            if tmp_index not in tmp_frame.index:
                tmp_frame.loc[tmp_index] = [0]

def merge_dataframe(*dataframe):
    i = 0
    for tmp_dataframe in dataframe:
        tmp_dataframe.sort_index(inplace=True)
        tmp_dataframe.column = str(i)
        i = i + 1
    new_dataframe = dataframe[0]
    i = 1
    while i < len(dataframe):
        new_dataframe = pd.merge(new_dataframe, dataframe[i], left_index=True, right_index=True)
        i = i + 1
    return new_dataframe

'''
A toolbox for simple comparison with natural
'''
#四个
def blastn_evaluation(file1_blast, file2_blast, file3_blast, rand_blast, report_path):
    '''
    1. Data loading
    '''
    file1_record = open_BLAST(file1_blast)
    min_num1, max_num1, density_natural = log_density(file1_record)

    file2_record = open_BLAST(file2_blast)
    min_num2, max_num2, density_gen = log_density(file2_record)

    file3_record = open_BLAST(file3_blast)
    min_num3, max_num3, density_rand = log_density(file3_record)

    rand_record = open_BLAST(rand_blast)
    min_num4, max_num4, density_extra = log_density(rand_record)

    max_num = max(max_num1, max_num2, max_num3, max_num4)
    min_num = min(min_num1, min_num2, min_num3, min_num4)

    '''
    2. plot density and bar plot
    '''
    font_label = {'family': 'Arial', 'weight': 'normal', 'size': 16}

    # draw the density and bar plot of blast results
    # plot_density(min_num, max_num, color=['chocolate', 'palegreen', 'lightblue', 'purple'],
    #              deepseed_promoters=density_natural, STR_sequences=density_gen, random_sequences=density_rand, extra_sequences=density_extra)
    plot_bar(min_num, max_num, color=['crimson','#1b9e77', '#d95f02', 'cornflowerblue'],
             STR_designed_promoters=file1_record, Alper_promoters=file2_record, iGEM_promoters=file3_record, random_sequences=rand_record)

    plt.xlabel('$\mathregular{log_{10}}$ $\it{e}$-value', font_label)
    plt.ylabel('Density', font_label)
    plt.savefig(report_path + 'blastn_evalue_barplot.png', bbox_inches='tight', dpi=900)

    '''
    3. transmit the format into Dataframe
    '''
    file1_fre = collections.Counter(file1_record)
    file2_fre = collections.Counter(file2_record)
    file3_fre = collections.Counter(file3_record)
    rand_fre = collections.Counter(rand_record)

    df1 = pd.DataFrame.from_dict(file1_fre, orient='index')
    df1 = df1 / sum(df1[0])
    df2 = pd.DataFrame.from_dict(file2_fre, orient='index')
    df2 = df2 / sum(df2[0])
    df3 = pd.DataFrame.from_dict(file3_fre, orient='index')
    df3 = df3 / sum(df3[0])
    df4 = pd.DataFrame.from_dict(rand_fre, orient='index')
    df4 = df4 / sum(df4[0])

    all_index = []
    find_index(all_index, df1, df2, df3, df4)
    add_index(all_index, df1, df2, df3, df4)
    df = merge_dataframe(df1, df2, df3, df4)
    df.columns = ['file1_record', 'file2_record', 'file3_record', 'random sequences']

    '''
    4. Draw the bar plot
    '''
    fig, ax, plt1, labels = set_default()
    df['file1_record'].plot(kind='bar', ax=ax, color='darkred', alpha=0.7, label='STR-designed promoters')
    df['file2_record'].plot(kind='bar', ax=ax, color='blue', alpha=0.5, label='Alper promoters')
    df['file3_record'].plot(kind='bar', ax=ax, color='orange', alpha=0.3, label='iGEM promoters')
    df['random sequences'].plot(kind='bar', ax=ax, color='indianred', alpha=0.3, label='Random sequences')

    plt1.legend(loc="best", prop={'family': 'Arial', 'size': 14})
    plt1.xlabel('$\it{e}$-value', font_label)
    plt1.ylabel('density', font_label)
    plt1.savefig(report_path + 'blastn_evalue_distribution.png', bbox_inches='tight', dpi=900)

    return

# 三个
# def blastn_evaluation( nat_blast, gen_blast, rand_blast, report_path):
#     '''
#     1. Data loading
#     '''
#     natural_record = open_BLAST(nat_blast)
#     min_num1, max_num1, density_natural = log_density(natural_record)
#
#     gen_record = open_BLAST(gen_blast)
#     min_num2, max_num2, density_gen = log_density(gen_record)
#
#     rand_record = open_BLAST(rand_blast)
#     min_num3, max_num3, density_rand = log_density(rand_record)
#
#     max_num = max(max_num1, max_num2, max_num3)
#     min_num = min(min_num1, min_num2, min_num3)
#
#     '''
#     2. plot density and bar plot
#     '''
#     font_label = {'family': 'Arial', 'weight': 'normal', 'size': 16}
#
#     # draw the density and bar plot of blast results
#     # plot_density(min_num, max_num, color=['chocolate', 'palegreen', 'lightblue'],
#     #              deepseed_promoters=density_natural, STR_sequences=density_gen, random_sequences=density_rand)
#     plot_bar(min_num, max_num, color=['#d95f02', '#7570b3', '#1b9e77'],
#              deepseed_promoters=natural_record, STR_sequences=gen_record, random_sequences=rand_record)
#
#     plt.xlabel('$\mathregular{log_{10}}$ $\it{e}$-value', font_label)
#     plt.ylabel('Density', font_label)
#     plt.savefig(report_path + 'blastn_evalue_barplot.png', bbox_inches='tight', dpi=900)
#
#     '''
#     3. transmit the format into Dataframe
#     '''
#     natural_fre = collections.Counter(natural_record)
#     gen_fre = collections.Counter(gen_record)
#     rand_fre = collections.Counter(rand_record)
#
#     df1 = pd.DataFrame.from_dict(natural_fre, orient='index')
#     df1 = df1 / sum(df1[0])
#     df2 = pd.DataFrame.from_dict(gen_fre, orient='index')
#     df2 = df2 / sum(df2[0])
#     df3 = pd.DataFrame.from_dict(rand_fre, orient='index')
#     df3 = df3 / sum(df3[0])
#
#     all_index = []
#     find_index(all_index, df1, df2, df3)
#     add_index(all_index, df1, df2, df3)
#     df = merge_dataframe(df1, df2, df3)
#     df.columns = ['natural promoters', 'gen sequences', 'random sequences']
#
#     '''
#     4. Draw the bar plot
#     '''
#     fig, ax, plt1, labels = set_default()
#     df['natural promoters'].plot(kind='bar', ax=ax, color='darkred', alpha=0.7, label='deepseed sequences')
#     df['gen sequences'].plot(kind='bar', ax=ax, color='orange', alpha=0.5, label='STR sequences')
#     df['random sequences'].plot(kind='bar', ax=ax, color='blue', alpha=0.3, label='random sequences')
#
#     plt1.legend(loc="best", prop={'family': 'Arial', 'size': 14})
#     plt1.xlabel('$\it{e}$-value', font_label)
#     plt1.ylabel('density', font_label)
#     plt1.savefig(report_path + 'blastn_evalue_distribution.png', bbox_inches='tight', dpi=900)
#
#     return

# 两个
# def blastn_evaluation(gen_blast, nat_blast, report_path):
#
#     '''
#     1. Data loading
#     '''
#
#     natural_record,natural_average_e_value = open_BLAST(nat_blast)
#     min_num1, max_num1, density_natural = log_density(natural_record)
#
#     gen_record,gen_average_e_value = open_BLAST(gen_blast)
#     min_num2, max_num2, density_gen = log_density(gen_record)
#
#     max_num = max(max_num1,max_num2)
#     min_num = min(min_num1,min_num2)
#
#     '''
#     2. plot density and bar plot
#     '''
#
#     font_label = {'family' : 'Arial', 'weight' : 'normal', 'size'   : 16}
#
#     # draw the density and bar plot of blast results
#     plot_density(min_num, max_num, color = ['chocolate','palegreen'], natural_promoters = density_natural, gen_sequences = density_gen)
#     plot_bar(min_num,max_num, color = ['#d95f02','#7570b3'], natural_promoters = natural_record, gen_sequences = gen_record)
#
#     plt.xlabel('$\mathregular{log_{10}}$ $\it{e}$-value', font_label)
#     plt.ylabel('Density',font_label)
#     plt.savefig(report_path + 'blastn_evalue_barplot.png',bbox_inches = 'tight',dpi=900)
#
#     '''
#     3. transmit the format into Dataframe
#     '''
#     natural_fre = collections.Counter(natural_record)
#     gen_fre = collections.Counter(gen_record)
#
#     df1 = pd.DataFrame.from_dict(natural_fre, orient='index')
#     df1 = df1/sum(df1[0])
#     df2 = pd.DataFrame.from_dict(gen_fre, orient='index')
#     df2 = df2/sum(df2[0])
#
#     all_index = []
#     find_index(all_index,df1,df2)
#     add_index(all_index,df1,df2)
#     df = merge_dataframe(df1,df2)
#     df.columns = ['natural promoters','gen sequences']
#
#     '''
#     4. Draw the bar plot
#     '''
#     fig,ax,plt1,labels = set_default()
#     df['natural promoters'].plot(kind='bar',ax=ax,color='darkred',alpha=0.7,label = 'natural promoters')
#     df['gen sequences'].plot(kind='bar',ax=ax,color='orange',alpha=0.5,label = 'gen sequences')
#
#     plt1.legend(loc="best",prop = {'family':'Arial','size':14})
#     plt1.xlabel('$\it{e}$-value',font_label)
#     plt1.ylabel('density',font_label)
#     plt1.savefig(report_path + 'blastn_evalue_distribution.png',bbox_inches = 'tight',dpi=900)
#
#     return